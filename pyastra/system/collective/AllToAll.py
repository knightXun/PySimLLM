# 假设 Ring 类、RingTopology 类、CallData 类、ComType 枚举、InjectionPolicy 枚举已经定义
# 这里简单假设 RingTopology 的 Dimension 是一个枚举
from enum import Enum


class Dimension(Enum):
    Local = 1


class ComType(Enum):
    All_to_All = 1
    All_Reduce = 2


class InjectionPolicy:
    pass


class Ring:
    def __init__(self, type, id, layer_num, logicalTopology, data_size, direction, injection_policy, boost_mode):
        self.type = type
        self.id = id
        self.layer_num = layer_num
        self.logicalTopology = logicalTopology
        self.data_size = data_size
        self.direction = direction
        self.injection_policy = injection_policy
        self.boost_mode = boost_mode
        self.nodes_in_ring = 0  # 这里假设 nodes_in_ring 在 Ring 类中有定义


class RingTopology:
    def __init__(self):
        self.dimension = Dimension.Local
        self.enabled = True

    def is_enabled(self):
        return self.enabled

    def get_receiver_node(self, node, direction):
        # 这里简单返回 node，实际应实现具体逻辑
        return node

    def get_sender_node(self, node, direction):
        # 这里简单返回 node，实际应实现具体逻辑
        return node


class CallData:
    pass


class EventType(Enum):
    General = 1
    PacketReceived = 2
    StreamInit = 3


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