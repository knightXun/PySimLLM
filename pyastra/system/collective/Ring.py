from enum import Enum
from collections import deque

# 假设这些类和枚举已经在其他地方定义
class ComType(Enum):
    All_Reduce = 1
    All_to_All = 2
    All_Gather = 3
    Reduce_Scatter = 4

class InjectionPolicy(Enum):
    Aggressive = 1
    Normal = 2

class StreamState(Enum):
    Created = 1
    Ready = 2
    Executing = 3
    Zombie = 4
    Dead = 5

class Callable:
    pass

class CallData:
    pass

class EventType(Enum):
    General = 1
    PacketReceived = 2
    StreamInit = 3

class Sys:
    dummy_data = None

    @staticmethod
    def handleEvent():
        pass

    @staticmethod
    def boostedTick():
        return 0

    @staticmethod
    def sys_panic(message):
        raise Exception(message)

class Stream:
    def __init__(self, owner, stream_num, current_queue_id):
        self.owner = owner
        self.stream_num = stream_num
        self.current_queue_id = current_queue_id
        self.state = StreamState.Created

    def changeState(self, new_state):
        self.state = new_state

class PacketBundle:
    def __init__(self, owner, stream, packets, processed, send_back, msg_size, transmition):
        self.owner = owner
        self.stream = stream
        self.packets = packets
        self.processed = processed
        self.send_back = send_back
        self.msg_size = msg_size
        self.transmition = transmition

    def send_to_MA(self):
        pass

    def send_to_NPU(self):
        pass

class MyPacket:
    def __init__(self, vnet, sender, receiver):
        self.vnet = vnet
        self.sender = sender
        self.preferred_dest = receiver
        self.preferred_src = sender
        self.preferred_vnet = vnet
        self.stream_num = 0

    def set_notifier(self, notifier):
        pass

class RingTopology:
    class Direction(Enum):
        Forward = 1
        Backward = 2

    class Dimension(Enum):
        Local = 1
        Global = 2

    def __init__(self, dimension):
        self.dimension = dimension
        self.nodes = []

    def get_nodes_in_ring(self):
        return len(self.nodes)

    def get_receiver_node(self, id, direction):
        return (id + 1) % len(self.nodes)

    def get_sender_node(self, id, direction):
        return (id - 1) % len(self.nodes)

    def is_enabled(self):
        return True

class Algorithm:
    def __init__(self, layer_num):
        self.layer_num = layer_num

class sim_request:
    def __init__(self):
        self.srcRank = 0
        self.dstRank = 0
        self.tag = 0
        self.reqType = 0
        self.vnet = 0
        self.layerNum = 0

class RecvPacketEventHadndlerData:
    def __init__(self, stream, id, event_type, vnet, stream_num):
        self.stream = stream
        self.id = id
        self.event_type = event_type
        self.vnet = vnet
        self.stream_num = stream_num

class Ring(Algorithm):
    g_ring_inCriticalSection = False

    class ringCriticalSection:
        def __init__(self):
            while Ring.g_ring_inCriticalSection:
                pass
            Ring.g_ring_inCriticalSection = True

        def ExitSection(self):
            Ring.g_ring_inCriticalSection = False

        def __del__(self):
            Ring.g_ring_inCriticalSection = False

    def __init__(self, type, id, layer_num, ring_topology, data_size, direction, injection_policy, boost_mode):
        super().__init__(layer_num)
        self.comType = type
        self.id = id
        self.logicalTopology = ring_topology
        self.data_size = data_size
        self.direction = direction
        self.nodes_in_ring = ring_topology.get_nodes_in_ring()
        self.current_receiver = ring_topology.get_receiver_node(id, direction)
        self.current_sender = ring_topology.get_sender_node(id, direction)
        self.parallel_reduce = 1
        self.injection_policy = injection_policy
        self.total_packets_sent = 0
        self.total_packets_received = 0
        self.free_packets = 0
        self.zero_latency_packets = 0
        self.non_zero_latency_packets = 0
        self.toggle = False
        self.name = "Ring"
        self.enabled = True
        if boost_mode:
            self.enabled = ring_topology.is_enabled()
        if ring_topology.dimension == RingTopology.Dimension.Local:
            self.transmition = "Fast"
        else:
            self.transmition = "Usual"
        if type == ComType.All_Reduce:
            self.stream_count = 2 * (self.nodes_in_ring - 1)
        elif type == ComType.All_to_All:
            self.stream_count = ((self.nodes_in_ring - 1) * self.nodes_in_ring) // 2
            if injection_policy == InjectionPolicy.Aggressive:
                self.parallel_reduce = self.nodes_in_ring - 1
            else:
                self.parallel_reduce = 1
        else:
            self.stream_count = self.nodes_in_ring - 1
        if type == ComType.All_to_All or type == ComType.All_Gather:
            self.max_count = 0
        else:
            self.max_count = self.nodes_in_ring - 1
        self.remained_packets_per_message = 1
        self.remained_packets_per_max_count = 1
        if type == ComType.All_Reduce:
            self.final_data_size = data_size
            self.msg_size = data_size // self.nodes_in_ring
        elif type == ComType.All_Gather:
            self.final_data_size = data_size * self.nodes_in_ring
            self.msg_size = data_size
        elif type == ComType.Reduce_Scatter:
            self.final_data_size = data_size // self.nodes_in_ring
            self.msg_size = data_size // self.nodes_in_ring
        elif type == ComType.All_to_All:
            self.final_data_size = data_size
            self.msg_size = data_size // self.nodes_in_ring
        self.packets = deque()
        self.locked_packets = []
        self.processed = False
        self.send_back = False
        self.NPU_to_MA = True

    def get_non_zero_latency_packets(self):
        return (self.nodes_in_ring - 1) * self.parallel_reduce * 1

    def run(self, event, data):
        if event == EventType.General:
            self.free_packets += 1
            self.ready()
            self.iteratable()
        elif event == EventType.PacketReceived:
            self.total_packets_received += 1
            self.insert_packet(None)
        elif event == EventType.StreamInit:
            for i in range(self.parallel_reduce):
                self.insert_packet(None)

    def release_packets(self):
        for packet in self.locked_packets:
            packet.set_notifier(self)
        if self.NPU_to_MA:
            PacketBundle(self.stream.owner, self.stream, self.locked_packets, self.processed, self.send_back, self.msg_size, self.transmition).send_to_MA()
        else:
            PacketBundle(self.stream.owner, self.stream, self.locked_packets, self.processed, self.send_back, self.msg_size, self.transmition).send_to_NPU()
        self.locked_packets = []

    def process_stream_count(self):
        if self.remained_packets_per_message > 0:
            self.remained_packets_per_message -= 1
        if self.id == 0:
            pass
        if self.remained_packets_per_message == 0 and self.stream_count > 0:
            self.stream_count -= 1
            if self.stream_count > 0:
                self.remained_packets_per_message = 1
        if self.remained_packets_per_message == 0 and self.stream_count == 0 and self.stream.state != StreamState.Dead:
            self.stream.changeState(StreamState.Zombie)

    def process_max_count(self):
        if self.remained_packets_per_max_count > 0:
            self.remained_packets_per_max_count -= 1
        if self.remained_packets_per_max_count == 0:
            self.max_count -= 1
            self.release_packets()
            self.remained_packets_per_max_count = 1

    def reduce(self):
        self.process_stream_count()
        self.packets.popleft()
        self.free_packets -= 1
        self.total_packets_sent += 1

    def iteratable(self):
        if self.stream_count == 0 and self.free_packets == (self.parallel_reduce * 1):
            self.exit()
            return False
        return True

    def insert_packet(self, sender):
        if not self.enabled:
            return
        if self.zero_latency_packets == 0 and self.non_zero_latency_packets == 0:
            self.zero_latency_packets = self.parallel_reduce * 1
            self.non_zero_latency_packets = self.get_non_zero_latency_packets()
            self.toggle = not self.toggle
        if self.zero_latency_packets > 0:
            packet = MyPacket(self.stream.current_queue_id, self.current_sender, self.current_receiver)
            packet.sender = sender
            self.packets.append(packet)
            self.locked_packets.append(packet)
            self.processed = False
            self.send_back = False
            self.NPU_to_MA = True
            self.process_max_count()
            self.zero_latency_packets -= 1
            return
        elif self.non_zero_latency_packets > 0:
            packet = MyPacket(self.stream.current_queue_id, self.current_sender, self.current_receiver)
            packet.sender = sender
            self.packets.append(packet)
            self.locked_packets.append(packet)
            if self.comType == ComType.Reduce_Scatter or (self.comType == ComType.All_Reduce and self.toggle):
                self.processed = True
            else:
                self.processed = False
            if self.non_zero_latency_packets <= self.parallel_reduce * 1:
                self.send_back = False
            else:
                self.send_back = True
            self.NPU_to_MA = False
            print(f"id: {self.id} non-zero latency packets at tick: {Sys.boostedTick()}")
            self.process_max_count()
            self.non_zero_latency_packets -= 1
            return
        Sys.sys_panic("should not inject nothing!")

    def ready(self):
        if self.stream.state == StreamState.Created or self.stream.state == StreamState.Ready:
            self.stream.changeState(StreamState.Executing)
        if not self.enabled or len(self.packets) == 0 or self.stream_count == 0 or self.free_packets == 0:
            return False
        packet = self.packets[0]
        snd_req = sim_request()
        snd_req.srcRank = self.id
        snd_req.dstRank = packet.preferred_dest
        snd_req.tag = self.stream.stream_num
        snd_req.reqType = 0
        snd_req.vnet = self.stream.current_queue_id
        snd_req.layerNum = self.layer_num
        self.stream.owner.front_end_sim_send(0, Sys.dummy_data, self.msg_size, 0, packet.preferred_dest, self.stream.stream_num, snd_req, Sys.handleEvent, None)
        rcv_req = sim_request()
        rcv_req.vnet = self.stream.current_queue_id
        rcv_req.layerNum = self.layer_num
        ehd = RecvPacketEventHadndlerData(self.stream, self.stream.owner.id, EventType.PacketReceived, packet.preferred_vnet, packet.stream_num)
        self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.msg_size, 0, packet.preferred_src, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)
        self.reduce()
        return True

    def exit(self):
        if len(self.packets) != 0:
            self.packets = deque()
        if len(self.locked_packets) != 0:
            self.locked_packets = []
        self.stream.owner.proceed_to_next_vnet_baseline(self.stream)