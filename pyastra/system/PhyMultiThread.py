import threading
import time
from dataclasses import dataclass
from collections import defaultdict
import logging
from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr
from pyverbs.cq import CQ, CompletionQueue
from pyverbs.addr import AH, AHAttr
from pyverbs.mr import MR
from pyverbs.enums import *
from pyverbs.wr import SendWR, RecvWR, SGE

# 日志配置
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("PhyMultiThread")

# 全局变量
all_recv_size = defaultdict(threading.Lock)  # 每个flow_id独立锁
all_send_size = defaultdict(threading.Lock)
end_flag = False
send_finished_callback = None
receive_finished_callback = None

@dataclass
class ncclFlowTag:
    channel_id: int
    chunk_id: int
    current_flow_id: int
    child_flow_id: int
    sender_node: int
    receiver_node: int
    flow_size: int
    pQps: int
    tag_id: int
    nvls_on: bool
    tree_flow_list: list = None

    def __post_init__(self):
        self.tree_flow_list = self.tree_flow_list or []

class TransportData:
    def __init__(self, **kwargs):
        self.channel_id = kwargs.get('channel_id', 0)
        self.chunk_id = kwargs.get('chunk_id', 0)
        self.current_flow_id = kwargs.get('current_flow_id', 0)
        self.child_flow_id = kwargs.get('child_flow_id', 0)
        self.sender_node = kwargs.get('sender_node', 0)
        self.receiver_node = kwargs.get('receiver_node', 0)
        self.flow_size = kwargs.get('flow_size', 0)
        self.pQps = kwargs.get('pQps', 0)
        self.tag_id = kwargs.get('tag_id', 0)
        self.nvls_on = kwargs.get('nvls_on', False)
        self.child_flow_list = kwargs.get('child_flow_list', [])
        self.child_flow_size = len(self.child_flow_list)

class PhyMtpInterface:
    class ExplicitCriticalSection:
        def __init__(self):
            self.lock = threading.Lock()
            self.lock.acquire()

        def exit_section(self):
            self.lock.release()

        def __del__(self):
            if self.lock.locked():
                self.lock.release()

    g_e_inCriticalSection = threading.Lock()

def set_send_finished_callback(handler):
    global send_finished_callback
    send_finished_callback = handler

def set_receive_finished_callback(handler):
    global receive_finished_callback
    receive_finished_callback = handler

def insert_recv_cqe(buff):
    ptrrecvdata: TransportData = buff
    flow_tag = ncclFlowTag(
        channel_id=ptrrecvdata.channel_id,
        chunk_id=ptrrecvdata.chunk_id,
        current_flow_id=ptrrecvdata.current_flow_id,
        child_flow_id=ptrrecvdata.child_flow_id,
        sender_node=ptrrecvdata.sender_node,
        receiver_node=ptrrecvdata.receiver_node,
        flow_size=ptrrecvdata.flow_size,
        pQps=ptrrecvdata.pQps,
        tag_id=ptrrecvdata.tag_id,
        nvls_on=ptrrecvdata.nvls_on
    )
    flow_tag.tree_flow_list = ptrrecvdata.child_flow_list.copy()
    logger.debug(f"PhyMultiThread::insert_recv_cqe src_id {flow_tag.sender_node} dst_id {flow_tag.receiver_node} flow_id {flow_tag.current_flow_id} channel_id {flow_tag.channel_id}")
    if receive_finished_callback:
        receive_finished_callback(flow_tag)

def insert_send_cqe(buff):
    ptrrecvdata: TransportData = buff
    flow_tag = ncclFlowTag(
        channel_id=ptrrecvdata.channel_id,
        chunk_id=ptrrecvdata.chunk_id,
        current_flow_id=ptrrecvdata.current_flow_id,
        child_flow_id=ptrrecvdata.child_flow_id,
        sender_node=ptrrecvdata.sender_node,
        receiver_node=ptrrecvdata.receiver_node,
        flow_size=ptrrecvdata.flow_size,
        pQps=ptrrecvdata.pQps,
        tag_id=ptrrecvdata.tag_id,
        nvls_on=ptrrecvdata.nvls_on
    )
    flow_tag.tree_flow_list = ptrrecvdata.child_flow_list.copy()
    logger.debug(f"PhyMultiThread::insert_send_cqe src_id {flow_tag.sender_node} dst_id {flow_tag.receiver_node} flow_id {flow_tag.current_flow_id} channel_id {flow_tag.channel_id}")
    if send_finished_callback:
        send_finished_callback(flow_tag)

def judge_polling_all_recv_cqe(buff):
    ptrrecvdata: TransportData = buff
    flow_id = ptrrecvdata.current_flow_id
    with all_recv_size[flow_id]:
        count = all_recv_size.get(flow_id, 0) + 1
        all_recv_size[flow_id] = count
        logger.debug(f"judge_polling_all_recv_cqe flow_id {flow_id} recv_cqe_size {count}")
        return count == NCCL_QPS_PER_PEER

def judge_polling_all_send_cqe(buff):
    ptrrecvdata: TransportData = buff
    flow_id = ptrrecvdata.current_flow_id
    with all_send_size[flow_id]:
        count = all_send_size.get(flow_id, 0) + 1
        all_send_size[flow_id] = count
        logger.debug(f"judge_polling_all_send_cqe flow_id {flow_id} send_cqe_size {count}")
        return count == NCCL_QPS_PER_PEER

class FlowPhyRdma:
    def __init__(self):
        self.recv_buff_map = dict()  # (qp_num, wr_id, imm_data) -> buff
        self.send_buff_map = dict()  # (qp_num, wr_id) -> buff

    def recv_wr_id_to_buff(self, qp_num, wr_id, imm_data):
        return self.recv_buff_map.get((qp_num, wr_id, imm_data), None)

    def send_wr_id_to_buff(self, qp_num, wr_id):
        return self.send_buff_map.get((qp_num, wr_id), None)

flow_rdma = FlowPhyRdma()

def create_polling_cqe_thread(cq_ptr, lcore_id=0):
    logger.debug("PhyMultiThread::create_polling_cqe_thread begin")
    cq: CompletionQueue = cq_ptr

    while not end_flag:
        try:
            # 轮询完成队列，最多获取TEST_IO_DEPTH个完成项
            wcs = cq.poll(TEST_IO_DEPTH)
            if wcs:
                logger.debug(f"PhyMultiThread::create_polling_send_cqe_thread cqe num {len(wcs)}")
                for wc in wcs:
                    if wc.status != IBV_WC_SUCCESS:
                        logger.error(f"wr's status is error {wc.status} opcode {wc.opcode}")
                        continue

                    if wc.opcode in (IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM):
                        now_us = int(time.time() * 1e6)
                        logger.debug(f"poll_recv_cqe qpn {wc.qp_num} wr_id {wc.wr_id} chunk_id {wc.imm_data} time {now_us}")
                        recv_buff = flow_rdma.recv_wr_id_to_buff(wc.qp_num, wc.wr_id, wc.imm_data)
                        if recv_buff and judge_polling_all_recv_cqe(recv_buff):
                            insert_recv_cqe(recv_buff)

                    elif wc.opcode == IBV_WC_RDMA_WRITE:
                        now_us = int(time.time() * 1e6)
                        logger.debug(f"poll_send_cqe qpn {wc.qp_num} wr_id {wc.wr_id} time {now_us}")
                        send_buff = flow_rdma.send_wr_id_to_buff(wc.qp_num, wc.wr_id)
                        if send_buff and judge_polling_all_send_cqe(send_buff):
                            insert_send_cqe(send_buff)
        except Exception as e:
            logger.error(f"Polling CQE failed: {str(e)}")
            break

def notify_all_thread_finished():
    global end_flag
    end_flag = True
    logger.debug("PhyMultiThread::notify_all_thread_finished end")
