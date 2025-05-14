#TODO: 增加新的
import math
import time
from enum import Enum
from threading import Condition, Lock, RLock
from typing import Dict, List, Tuple, Optional, Any
import heapq
from datetime import datetime
from collections import defaultdict, deque

# 假设存在的基础类 (需要根据实际项目补充实现)
class Algorithm:
    def __init__(self, layer_num: int):
        self.layer_num = layer_num
        self.stream = None  # 类型应为Stream对象
        self.name = None
        self.enabled = True
        self.comType = None
        self.final_data_size = 0
        self.logicalTopology = None
        self.data_size = 0

class MyPacket:
    def __init__(self, vnet_id: int, sender: int, receiver: int, msg_size: int, channel_id: int, flow_id: int):
        self.vnet_id = vnet_id
        self.sender = sender
        self.receiver = receiver
        self.msg_size = msg_size
        self.channel_id = channel_id
        self.flow_id = flow_id
        self.preferred_dest = receiver
        self.preferred_vnet = vnet_id

class MockNcclLog:
    _instance = None
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
    
    def write_log(self, level: 'NcclLogLevel', message: str, *args):
        print(f"[{level.name}] {message % args}")

class NcclLogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

class Sys:
    dummy_data = bytes()
    
    @staticmethod
    def handleEvent(event_type: 'EventType', data: Any):
        pass

    @staticmethod
    def sys_panic(msg: str):
        raise RuntimeError(msg)

class StreamState(Enum):
    Created = 0
    Ready = 1
    Executing = 2
    Zombie = 3
    Dead = 4

class ComType(Enum):
    All_Reduce = 0
    All_Gather = 1
    Reduce_Scatter = 2
    All_to_All = 3
    All_Reduce_NVLS = 4

class EventType(Enum):
    General = 0
    PacketReceived = 1
    StreamInit = 2
    PacketSentFinshed = 3

class MemBus:
    class Transmition(Enum):
        Fast = 0
        Usual = 1

class RingTopology:
    class Dimension(Enum):
        Local = 0
        Remote = 1

    def __init__(self, dimension):
        self.dimension = dimension
    
    def get_nodes_in_ring(self) -> int:
        return 8  # 示例值

class MockNccl:
    class SingleFlow:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    FlowModels = Dict[Tuple[int, int], SingleFlow]

class ncclFlowTag:
    def __init__(self):
        self.current_flow_id = -1
        self.channel_id = -1
        self.sender_node = -1
        self.receiver_node = -1
        self.tree_flow_list = []
        self.tag_id = 0
        self.flow_size = 0
        self.chunk_id = 0

class sim_request:
    def __init__(self):
        self.vnet = -1
        self.layerNum = -1
        self.reqCount = 0
        self.tag = -1
        self.reqType = None
        self.srcRank = -1
        self.dstRank = -1
        self.flowTag = ncclFlowTag()

class PacketBundle:
    def __init__(self, owner, stream, packets, processed, send_back, size, transmition, channel_id, flow_id):
        pass
    
    def send_to_MA(self):
        pass
    
    def send_to_NPU(self):
        pass

# 主要类实现
class NcclTreeFlowModel(Algorithm):
    g_flow_inCriticalSection = False
    _flow_in_critical_lock = Lock()

    class FlowCriticalSection:
        def __enter__(self):
            while NcclTreeFlowModel.g_flow_inCriticalSection:
                pass
            NcclTreeFlowModel.g_flow_inCriticalSection = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            NcclTreeFlowModel.g_flow_inCriticalSection = False

    def __init__(self, 
                com_type: ComType,
                node_id: int,
                layer_num: int,
                ring_topology: RingTopology,
                data_size: int,
                direction: Any,  # RingTopology.Direction类型需要具体定义
                injection_policy: Any,
                boost_mode: bool,
                ptr_flow_models: Optional[Dict[Tuple[int, int], MockNccl.SingleFlow]],
                tree_channels: int):
        super().__init__(layer_num)
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.comType = com_type
        self.id = node_id
        self.logicalTopology = ring_topology
        self.data_size = data_size
        self.nodes_in_ring = ring_topology.get_nodes_in_ring()
        self.parallel_reduce = 1
        self.toggle = False
        self.name = "Ring"
        self.enabled = True
        self.m_channels = tree_channels
        self.judge_exit_flag = False
        self.send_packets = 0
        self.recv_packets = 0
        self.pQps = {}
        self.zero_latency_packets = defaultdict(int)
        self.non_zero_latency_packets = defaultdict(int)
        self._flow_models: Dict[Tuple[int, int], MockNccl.SingleFlow] = {}
        self.free_packets = defaultdict(int)
        self.indegree_mapping = defaultdict(int)
        self.packets = defaultdict(deque)
        self.transmition = MemBus.Transmition.Fast if ring_topology.dimension == RingTopology.Dimension.Local else MemBus.Transmition.Usual

        if boost_mode:
            self.enabled = ring_topology.is_enabled()  # 需要实现is_enabled方法

        if ptr_flow_models:
            for key, flow in ptr_flow_models.items():
                if flow.dest == self.id:
                    self.free_packets[(flow.channel_id, flow.src)] += 1
                    self._flow_models[key] = flow
                    self.recv_packets += 1
                if flow.src == self.id:
                    self._stream_count[flow.channel_id] += 1
                    self._flow_models[key] = flow
                    self.send_packets += 1

        for channel_id in range(self.m_channels):
            self.zero_latency_packets[channel_id] = 0
            self.non_zero_latency_packets[channel_id] = 0

        self.init_indegree_mapping()

        # 根据通信类型设置最终数据大小
        if com_type == ComType.All_Reduce:
            self.final_data_size = data_size
        elif com_type == ComType.All_Gather:
            self.final_data_size = data_size * self.nodes_in_ring
        elif com_type == ComType.Reduce_Scatter:
            self.final_data_size = data_size // self.nodes_in_ring
        elif com_type == ComType.All_to_All:
            self.final_data_size = data_size

    def init_indegree_mapping(self):
        for (channel_id, flow_id), flow in self._flow_models.items():
            if flow.src != self.id:
                continue
            self.indegree_mapping[flow_id] = len(flow.parent_flow_id)

    def run(self, event: EventType, data: Any):
        if event == EventType.General:
            self.handle_general_event(data)
        elif event == EventType.PacketReceived:
            self.handle_packet_received(data)
        elif event == EventType.StreamInit:
            self.handle_stream_init()
        elif event == EventType.PacketSentFinshed:
            self.handle_packet_sent(data)

    def handle_general_event(self, data: Any):
        channel_id = data.channel_id
        flow_id = data.flow_id
        self.ready(channel_id, flow_id)

    def handle_packet_received(self, data: Any):
        flow_tag = data.flowTag
        channel_id = flow_tag.channel_id
        self.recv_packets -= 1

        # 简化的处理逻辑
        with self.FlowCriticalSection():
            self.free_packets[(channel_id, flow_tag.sender_node)] -= 1

        for next_flow_id in flow_tag.tree_flow_list:
            if self.indegree_mapping[next_flow_id] > 0:
                self.indegree_mapping[next_flow_id] -= 1
                if self.indegree_mapping[next_flow_id] == 0:
                    self.ready(channel_id, next_flow_id)

    def handle_stream_init(self):
        # 简化的初始化流程
        for channel_id in range(self.m_channels):
            for flow in self._flow_models.values():
                if flow.src == self.id and flow.channel_id == channel_id and flow.chunk_id == 0:
                    self.insert_packets(channel_id, flow.flow_id)

    def handle_packet_sent(self, data: Any):
        flow_tag = data.flowTag
        channel_id = flow_tag.channel_id
        self.reduce(channel_id, flow_tag.current_flow_id)
        self.iteratable(channel_id)

    def insert_packets(self, channel_id: int, flow_id: int):
        flow = self._flow_models[(channel_id, flow_id)]
        # 简化的数据包插入逻辑
        packet = MyPacket(
            self.stream.current_queue_id,
            flow.src,
            flow.dest,
            flow.flow_size,
            channel_id,
            flow_id
        )
        self.packets[(channel_id, flow_id)].append(packet)
        self.send_packet(packet)

    def send_packet(self, packet: MyPacket):
        # 简化的发送逻辑
        send_ehd = SendPacketEventHandlerData(...)
        self.stream.owner.front_end_sim_send(
            0, Sys.dummy_data, packet.msg_size, 
            'UINT8', packet.receiver, packet.channel_id, 
            None, Sys.handleEvent, send_ehd
        )

    # 其他方法需要根据实际项目需求补充实现

# 辅助类和函数需要根据项目实际情况补充