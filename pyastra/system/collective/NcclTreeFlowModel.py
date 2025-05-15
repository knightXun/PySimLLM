#TODO: 增加新的
import math
import time
from enum import Enum
from threading import Condition, Lock, RLock
from typing import Dict, List, Tuple, Optional, Any
import heapq
from datetime import datetime
from collections import defaultdict, deque


from MemBus import MemBus
from MyPacket import MyPacket
from Algorithm import Algorithm
from MockNcclLog import MockNcclLog, NcclLogLevel
from Sys import Sys
from StreamStat import StreamStat
from Common import ComType, EventType, StreamState

from system.topology.RingTopology import RingTopology
from AstraNetworkAPI import sim_request
from system.MockNcclChannel import *
from system.SendPacketEventHandlerData import SendPacketEventHandlerData
from PacketBundle import PacketBundle


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
                direction: Any, 
                injection_policy: Any,
                boost_mode: bool,
                ptr_flow_models: Optional[Dict[Tuple[int, int], SingleFlow]],
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
        self._flow_models: Dict[Tuple[int, int], SingleFlow] = {}
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