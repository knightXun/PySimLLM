# ---------------------- 测试代码 ----------------------
import unittest
from unittest.mock import Mock, MagicMock, patch

class TestPhyMultiThread(unittest.TestCase):
    def setUp(self):
        global end_flag, all_recv_size, all_send_size
        end_flag = False
        all_recv_size.clear()
        all_send_size.clear()
        self.test_flow_id = 1001
        self.test_qp_num = 1234
        self.test_wr_id = 5678
        self.test_imm_data = 99

        # 创建模拟的TransportData
        self.test_recv_data = TransportData(
            current_flow_id=self.test_flow_id,
            sender_node=1,
            receiver_node=2,
            child_flow_list=[101, 102],
            channel_id=0,
            chunk_id=1
        )
        self.test_send_data = TransportData(
            current_flow_id=self.test_flow_id,
            sender_node=2,
            receiver_node=1,
            child_flow_list=[201, 202],
            channel_id=0,
            chunk_id=1
        )

        # 模拟RDMA完成队列
        self.mock_cq = Mock(spec=CompletionQueue)
        self.mock_wc_recv = Mock()
        self.mock_wc_recv.status = IBV_WC_SUCCESS
        self.mock_wc_recv.opcode = IBV_WC_RECV
        self.mock_wc_recv.qp_num = self.test_qp_num
        self.mock_wc_recv.wr_id = self.test_wr_id
        self.mock_wc_recv.imm_data = self.test_imm_data

        self.mock_wc_send = Mock()
        self.mock_wc_send.status = IBV_WC_SUCCESS
        self.mock_wc_send.opcode = IBV_WC_RDMA_WRITE
        self.mock_wc_send.qp_num = self.test_qp_num
        self.mock_wc_send.wr_id = self.test_wr_id

    def test_receive_callback_trigger(self):
        """测试接收完成回调触发逻辑"""
        # 准备数据映射
        flow_rdma.recv_buff_map[(self.test_qp_num, self.test_wr_id, self.test_imm_data)] = self.test_recv_data
        
        # 设置回调记录器
        callback_called = False
        def test_receive_handler(flow_tag):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(flow_tag.current_flow_id, self.test_flow_id)
            self.assertEqual(flow_tag.tree_flow_list, [101, 102])
        set_receive_finished_callback(test_receive_handler)

        # 模拟第一次轮询（不触发）
        self.mock_cq.poll.return_value = [self.mock_wc_recv]
        create_polling_cqe_thread(self.mock_cq)  # 单次调用模拟轮询
        self.assertFalse(callback_called)
        self.assertEqual(all_recv_size[self.test_flow_id], 1)

        # 模拟第二次轮询（触发）
        create_polling_cqe_thread(self.mock_cq)
        self.assertTrue(callback_called)
        self.assertEqual(all_recv_size[self.test_flow_id], 2)

    def test_send_callback_trigger(self):
        """测试发送完成回调触发逻辑"""
        flow_rdma.send_buff_map[(self.test_qp_num, self.test_wr_id)] = self.test_send_data
        
        callback_called = False
        def test_send_handler(flow_tag):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(flow_tag.current_flow_id, self.test_flow_id)
            self.assertEqual(flow_tag.tree_flow_list, [201, 202])
        set_send_finished_callback(test_send_handler)

        # 第一次轮询（不触发）
        self.mock_cq.poll.return_value = [self.mock_wc_send]
        create_polling_cqe_thread(self.mock_cq)
        self.assertFalse(callback_called)
        self.assertEqual(all_send_size[self.test_flow_id], 1)

        # 第二次轮询（触发）
        create_polling_cqe_thread(self.mock_cq)
        self.assertTrue(callback_called)
        self.assertEqual(all_send_size[self.test_flow_id], 2)

    def test_thread_safety(self):
        """测试计数器的线程安全性"""
        test_flow_id = 9999
        test_data = TransportData(current_flow_id=test_flow_id)
        flow_rdma.recv_buff_map[(self.test_qp_num, self.test_wr_id, self.test_imm_data)] = test_data

        def worker():
            for _ in range(NCCL_QPS_PER_PEER):
                self.mock_cq.poll.return_value = [self.mock_wc_recv]
                create_polling_cqe_thread(self.mock_cq)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()

        self.assertEqual(all_recv_size[test_flow_id], 3 * NCCL_QPS_PER_PEER)

    def test_thread_termination(self):
        """测试轮询线程终止逻辑"""
        global end_flag
        thread = threading.Thread(target=create_polling_cqe_thread, args=(self.mock_cq,))
        thread.start()
        time.sleep(0.1)  # 等待线程启动
        self.assertTrue(thread.is_alive())

        notify_all_thread_finished()
        thread.join(1)  # 等待线程终止
        self.assertFalse(thread.is_alive())

if __name__ == '__main__':
    unittest.main()