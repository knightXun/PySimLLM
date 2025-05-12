import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

# 模拟AstraSim命名空间
class AstraSim:
    class EventType:
        PacketReceived = "PacketReceived"
        PacketSentFinshed = "PacketSentFinshed"

    @dataclass
    class RecvPacketEventHadndlerData:
        owner: Any
        event_type: EventType
        flow_tag: Any

    @dataclass
    class SendPacketEventHandlerData:
        owner: Any
        sender_node: int
        receiver_node: int
        channel_id: int
        event_type: EventType
        flow_tag: Any = None

    class StreamBaseline:
        def consume(self, event_data: RecvPacketEventHadndlerData) -> None:
            pass  # 实际实现需要补充
        
        def sendcallback(self, event_data: SendPacketEventHandlerData) -> None:
            pass  # 实际实现需要补充

    @dataclass
    class sim_request:
        flowTag: Any  # 对应ncclFlowTag类型

# 模拟ncclFlowTag结构
@dataclass
class ncclFlowTag:
    sender_node: int
    receiver_node: int
    channel_id: int
    flow_size: int
    current_flow_id: int
    tag_id: int
    chunk_id: int = 0
    child_flow_id: int = 0
    pQps: Any = None
    nvls_on: bool = False
    tree_flow_list: list = None

# 模拟TransportData结构
@dataclass
class TransportData:
    channel_id: int
    chunk_id: int
    current_flow_id: int
    child_flow_id: int
    sender_node: int
    receiver_node: int
    flow_size: int
    pQps: Any
    tag_id: int
    nvls_on: bool
    child_flow_size: int = 0
    child_flow_list: list = None

# 模拟MockNcclLog单例类
class MockNcclLog:
    _instance: Optional["MockNcclLog"] = None
    _log_levels = {
        "DEBUG": "DEBUG",
        # 可扩展其他日志级别
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def write_log(self, level: str, message: str, *args) -> None:
        formatted_msg = message % args
        print(f"[{level}] {formatted_msg}")

# 全局变量模拟
global_sys: Optional[AstraSim.StreamBaseline] = None
PHY_RDMA = True  # 可根据实际环境调整

# 模拟网络回调函数（需要外部实现）
_receive_finished_callback: Optional[Callable[[ncclFlowTag], None]] = None
_send_finished_callback: Optional[Callable[[ncclFlowTag], None]] = None

def set_receive_finished_callback(callback: Callable[[ncclFlowTag], None]) -> None:
    global _receive_finished_callback
    _receive_finished_callback = callback

def set_send_finished_callback(callback: Callable[[ncclFlowTag], None]) -> None:
    global _send_finished_callback
    _send_finished_callback = callback

# 核心功能函数
def notify_receiver_receive_data(sender_node: int, receiver_node: int, 
                                 message_size: int, flow_tag: ncclFlowTag) -> None:
    if global_sys and global_sys.running_list:  # 假设running_list是可迭代对象
        owner = global_sys.running_list[0]
        ehd = AstraSim.RecvPacketEventHadndlerData(
            owner=owner,
            event_type=AstraSim.EventType.PacketReceived,
            flow_tag=flow_tag
        )
        owner.consume(ehd)

def notify_sender_sending_finished(sender_node: int, receiver_node: int,
                                  message_size: int, flow_tag: ncclFlowTag) -> None:
    nccl_log = MockNcclLog()
    if global_sys and global_sys.running_list:
        owner = global_sys.running_list[0]
        send_ehd = AstraSim.SendPacketEventHandlerData(
            owner=owner,
            sender_node=flow_tag.sender_node,
            receiver_node=flow_tag.receiver_node,
            channel_id=flow_tag.channel_id,
            event_type=AstraSim.EventType.PacketSentFinshed,
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
    nccl_log = MockNcclLog()
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
              request: AstraSim.sim_request) -> None:
    nccl_log = MockNcclLog()
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
        # 模拟flow_rdma的simai_ibv_post_send方法（需要实际实现）
        class FlowPhyRdma:
            def simai_ibv_post_send(self, tag: int, src: int, dst: int, 
                                   send_data: TransportData, data_size: int, 
                                   max_packets: int, chunk_id: int) -> None:
                pass  # 实际实现需要补充
        
        flow_rdma = FlowPhyRdma()  # 假设flow_rdma是单例或预初始化对象
        flow_rdma.simai_ibv_post_send(
            tag,
            src,
            dst,
            send_data,
            sizeof(send_data),  # 注意：Python中没有sizeof，需根据实际情况处理
            max_packet_count,
            flow_tag.chunk_id
        )

# 示例用法（可根据需要调整）
if __name__ == "__main__":
    # 初始化全局系统对象
    global_sys = AstraSim.StreamBaseline()
    global_sys.running_list = [AstraSim.StreamBaseline()]  # 模拟running_list
    
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
    sample_request = AstraSim.sim_request(flowTag=sample_flow_tag)
    
    send_flow(
        src=0,
        dst=1,
        max_packet_count=10,
        msg_handler=lambda x: print(f"处理消息: {x}"),
        fun_arg=None,
        tag=0,
        request=sample_request
    )
    