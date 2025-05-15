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

from SimAiFlowModelRdma import FlowPhyRdma
import MockNcclLog
from MockNcclLog import NcclLogLevel
from AstraNetworkAPI import ncclFlowTag
from SimAiPhyCommon import *

all_recv_size = defaultdict(threading.Lock) 
all_send_size = defaultdict(threading.Lock)
end_flag = False
send_finished_callback = None
receive_finished_callback = None


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
    logger = MockNcclLog.get_instance()

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

    logger.write_log(NcclLogLevel.DEBUG,f"PhyMultiThread::insert_recv_cqe src_id {flow_tag.sender_node} dst_id {flow_tag.receiver_node} flow_id {flow_tag.current_flow_id} channel_id {flow_tag.channel_id}")
    if receive_finished_callback:
        receive_finished_callback(flow_tag)


def insert_send_cqe(buff):
    logger = MockNcclLog.get_instance()

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
    logger = MockNcclLog.get_instance()

    ptrrecvdata: TransportData = buff
    flow_id = ptrrecvdata.current_flow_id
    with all_recv_size[flow_id]:
        count = all_recv_size.get(flow_id, 0) + 1
        all_recv_size[flow_id] = count
        logger.write_log(NcclLogLevel.DEBUG, f"judge_polling_all_recv_cqe flow_id {flow_id} recv_cqe_size {count}")
        return count == NCCL_QPS_PER_PEER

def judge_polling_all_send_cqe(buff):
    logger = MockNcclLog.get_instance()

    ptrrecvdata: TransportData = buff
    flow_id = ptrrecvdata.current_flow_id
    with all_send_size[flow_id]:
        count = all_send_size.get(flow_id, 0) + 1
        all_send_size[flow_id] = count
        logger.write_log(NcclLogLevel.DEBUG, f"judge_polling_all_send_cqe flow_id {flow_id} send_cqe_size {count}")
        return count == NCCL_QPS_PER_PEER

def create_polling_cqe_thread(cq_ptr, lcore_id=0):
    logger = MockNcclLog.get_instance()

    logger.write_log(NcclLogLevel.DEBUG, "PhyMultiThread::create_polling_cqe_thread begin")
    cq: CompletionQueue = cq_ptr

    while not end_flag:
        try:
            wcs = cq.poll(TEST_IO_DEPTH)
            if wcs:
                logger.write_log(NcclLogLevel.DEBUG, f"PhyMultiThread::create_polling_send_cqe_thread cqe num {len(wcs)}")
                for wc in wcs:
                    if wc.status != IBV_WC_SUCCESS:
                        logger.write_log(NcclLogLevel.ERROR, f"wr's status is error {wc.status} opcode {wc.opcode}")
                        continue

                    if wc.opcode in (IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM):
                        now_us = int(time.time() * 1e6)
                        logger.write_log(NcclLogLevel.DEBUG, f"poll_recv_cqe qpn {wc.qp_num} wr_id {wc.wr_id} chunk_id {wc.imm_data} time {now_us}")
                        recv_buff = FlowPhyRdma.flow_rdma.recv_wr_id_to_buff(wc.qp_num, wc.wr_id, wc.imm_data)
                        if recv_buff and judge_polling_all_recv_cqe(recv_buff):
                            insert_recv_cqe(recv_buff)

                    elif wc.opcode == IBV_WC_RDMA_WRITE:
                        now_us = int(time.time() * 1e6)
                        logger.write_log(NcclLogLevel.DEBUG, f"poll_send_cqe qpn {wc.qp_num} wr_id {wc.wr_id} time {now_us}")
                        send_buff = FlowPhyRdma.flow_rdma.send_wr_id_to_buff(wc.qp_num, wc.wr_id)
                        if send_buff and judge_polling_all_send_cqe(send_buff):
                            insert_send_cqe(send_buff)
        except Exception as e:
            logger.write_log(NcclLogLevel.ERROR, f"Polling CQE failed: {str(e)}")
            break

def notify_all_thread_finished():
    logger = MockNcclLog.get_instance()

    global end_flag
    end_flag = True
    logger.write_log(NcclLogLevel.DEBUG, "PhyMultiThread::notify_all_thread_finished end")
