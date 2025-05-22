import time, sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from system.MockNcclLog import MockNcclLog
from system.SimAiFlowModelRdma import flow_rdma
from system.PhyMultiThread import FlowPhyRdma, set_send_finished_callback, \
    set_receive_finished_callback, send_finished_callback, receive_finished_callback

from system.RecvPacketEventHandlerData import RecvPacketEventHandlerData
from system.SendPacketEventHandlerData import SendPacketEventHandlerData
from system.Common import EventType
from system.BaseStream import BaseStream
from system.StreamBaseline import StreamBaseline
from system.AstraNetworkAPI import sim_request, ncclFlowTag
from system.Sys import global_sys
from system.SimAiPhyCommon import TransportData


def notify_receiver_receive_data(sender_node: int, receiver_node: int, 
                                 message_size: int, flow_tag: ncclFlowTag) -> None:
    owner = global_sys.running_list[0]
    ehd = RecvPacketEventHandlerData(
            owner=owner,
            event_type= EventType.PacketReceived,
            flow_tag=flow_tag
        )
    owner.consume(ehd)


def notify_sender_sending_finished(sender_node: int, receiver_node: int,
                                  message_size: int, flow_tag: ncclFlowTag) -> None:
    nccl_log = MockNcclLog.get_instance()
    owner = global_sys.running_list[0]
    send_ehd = SendPacketEventHandlerData(
        owner=owner,
        sender_node=flow_tag.sender_node,
        receiver_node=flow_tag.receiver_node,
        channel_id=flow_tag.channel_id,
        event_type= EventType.PacketSentFinshed,
        flow_tag=flow_tag
    )

    nccl_log.write_log(
        "DEBUG",
        "notify_sender_sending_finished_test src %d dst %d channel_id %d flow_id %d",
        flow_tag.sender_node, flow_tag.receiver_node, 
        flow_tag.channel_id, flow_tag.channel_id
    )
    owner.sendcallback(send_ehd)

def simai_recv_finish(flow_tag: ncclFlowTag) -> None:
    sid = flow_tag.sender_node
    did = flow_tag.receiver_node
    notify_size = flow_tag.flow_size
    notify_receiver_receive_data(sid, did, notify_size, flow_tag)

def simai_send_finish(flow_tag: ncclFlowTag) -> None:
    sid = flow_tag.sender_node
    did = flow_tag.receiver_node
    nccl_log = MockNcclLog.get_instance()
    nccl_log.write_log(
        "DEBUG",
        "数据包出网卡队列, src %d did %d total_bytes %d channel_id %d flow_id %d tag_id %d",
        sid, did, flow_tag.flow_size, flow_tag.channel_id,
        flow_tag.current_flow_id, flow_tag.tag_id
    )
    notify_sender_sending_finished(sid, did, flow_tag.flow_size, flow_tag)

def set_simai_network_callback() -> None:
    set_receive_finished_callback(simai_recv_finish)
    set_send_finished_callback(simai_send_finish)

def send_flow(src: int, dst: int, max_packet_count: int,
              msg_handler: Callable[[Any], None], fun_arg: Any, tag: int, 
              request: sim_request) -> None:
    nccl_log = MockNcclLog.get_instance()
    flow_tag = request.flowTag
    
    send_data = TransportData(
        channel_id=flow_tag.channel_id,
        chunk_id=flow_tag.chunk_id,
        current_flow_id=flow_tag.current_flow_id,
        child_flow_id=flow_tag.child_flow_id,
        sender_node=flow_tag.sender_node,
        receiver_node=flow_tag.receiver_node,
        flow_size=flow_tag.flow_size,
        pQps=flow_tag.pQps,
        tag_id=flow_tag.tag_id,
        nvls_on=flow_tag.nvls_on
    )
    
    if flow_tag.tree_flow_list:
        send_data.child_flow_size = len(flow_tag.tree_flow_list)
        send_data.child_flow_list = flow_tag.tree_flow_list.copy()
    else:
        send_data.child_flow_size = 0
        send_data.child_flow_list = []
    
    nccl_log.write_log(
        "DEBUG",
        "SendPackets %d SendFlow to %d channelid: %d flow_id: %d size: %d tag_id %d",
        src, dst, tag, flow_tag.current_flow_id, max_packet_count, flow_tag.tag_id
    )
    
    if PHY_RDMA:
        
        flow_rdma.simai_ibv_post_send(
            tag,
            src,
            dst,
            send_data,
            sys.getsizeof(send_data),  # sys.getsizeof 替代 sizeof 操作符
            max_packet_count,
            flow_tag.chunk_id
        )

# local test case
if __name__ == "__main__":
    # 初始化全局系统对象
    global_sys = StreamBaseline()
    global_sys.running_list = [StreamBaseline()]  # 模拟running_list
    
    # 设置网络回调
    set_simai_network_callback()
    
    # 示例发送请求
    sample_flow_tag = ncclFlowTag(
        sender_node=0,
        receiver_node=1,
        channel_id=0,
        flow_size=1024,
        current_flow_id=1,
        tag_id=1001
    )
    sample_request = sim_request(flowTag=sample_flow_tag)
    
    send_flow(
        src=0,
        dst=1,
        max_packet_count=10,
        msg_handler=lambda x: print(f"处理消息: {x}"),
        fun_arg=None,
        tag=0,
        request=sample_request
    )
    