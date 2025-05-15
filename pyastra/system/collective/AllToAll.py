from system.topology.RingTopology import RingTopology
from AstraNetworkAPI import sim_request
from system.MockNcclChannel import *
from system.SendPacketEventHandlerData import SendPacketEventHandlerData
from PacketBundle import PacketBundle
from system.RecvPacketEventHandlerData import RecvPacketEventHandlerData
from Algorithm import Algorithm
from system.Common import ComType, EventType
from system.topology.BinaryTree import BinaryTree
from Ring import Ring


class AllToAll(Ring):
    def __init__(self, type, window, id, layer_num, allToAllTopology, data_size, direction, injection_policy, boost_mode):
        super().__init__(type, id, layer_num, allToAllTopology, data_size, direction, injection_policy, boost_mode)
        self.name = "AllToAll"
        self.enabled = True
        self.middle_point = self.nodes_in_ring - 1
        if boost_mode:
            self.enabled = allToAllTopology.is_enabled()
        if window == -1:
            self.parallel_reduce = self.nodes_in_ring - 1
        else:
            self.parallel_reduce = min(window, self.nodes_in_ring - 1)
        if type == ComType.All_to_All:
            self.stream_count = self.nodes_in_ring - 1
        self.remained_packets_per_max_count = 1
        self.max_count = 0
        self.total_packets_received = 0
        self.free_packets = 0
        self.current_receiver = id
        self.current_sender = id

    def get_non_zero_latency_packets(self):
        if self.logicalTopology.dimension != Dimension.Local:
            return self.parallel_reduce * 1
        else:
            return (self.nodes_in_ring - 1) * self.parallel_reduce * 1

    def process_max_count(self):
        if self.remained_packets_per_max_count > 0:
            self.remained_packets_per_max_count -= 1
        if self.remained_packets_per_max_count == 0:
            self.max_count -= 1
            self.release_packets()
            self.remained_packets_per_max_count = 1
            self.current_receiver = self.logicalTopology.get_receiver_node(self.current_receiver, self.direction)
            if self.current_receiver == self.id:
                self.current_receiver = self.logicalTopology.get_receiver_node(self.current_receiver, self.direction)
            self.current_sender = self.logicalTopology.get_sender_node(self.current_sender, self.direction)
            if self.current_sender == self.id:
                self.current_sender = self.logicalTopology.get_sender_node(self.current_sender, self.direction)

    def release_packets(self):
        # 这里需要实现具体的释放数据包逻辑
        pass

    def ready(self):
        # 这里需要实现具体的准备逻辑
        pass

    def iteratable(self):
        # 这里需要实现具体的迭代逻辑
        pass

    def insert_packet(self, packet):
        # 这里需要实现具体的插入数据包逻辑
        pass

    def run(self, event, data):
        if event == EventType.General:
            self.free_packets += 1
            if self.type == ComType.All_Reduce and self.stream_count <= self.middle_point:
                if self.total_packets_received < self.middle_point:
                    return
                for i in range(self.parallel_reduce):
                    self.ready()
                self.iteratable()
            else:
                self.ready()
                self.iteratable()
        elif event == EventType.PacketReceived:
            self.total_packets_received += 1
            self.insert_packet(None)
        elif event == EventType.StreamInit:
            for i in range(self.parallel_reduce):
                self.insert_packet(None)