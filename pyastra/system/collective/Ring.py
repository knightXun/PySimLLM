from enum import Enum
from collections import deque

from Callable import Callable
from CallData import CallData
from Sys import Sys 
from StreamStat import StreamStat
from Common import EventType, ComType, InjectionPolicy, StreamState
from MyPacket import MyPacket
from Algorithm import Algorithm
from PacketBundle import PacketBundle
from RecvPacketEventHandlerData import RecvPacketEventHandlerData
from system.topology.RingTopology import RingTopology
from AstraNetworkAPI import sim_request
from RecvPacketEventHandlerData import RecvPacketEventHandlerData

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
        ehd = RecvPacketEventHandlerData(self.stream, self.stream.owner.id, EventType.PacketReceived, packet.preferred_vnet, packet.stream_num)
        self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.msg_size, 0, packet.preferred_src, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)
        self.reduce()
        return True

    def exit(self):
        if len(self.packets) != 0:
            self.packets = deque()
        if len(self.locked_packets) != 0:
            self.locked_packets = []
        self.stream.owner.proceed_to_next_vnet_baseline(self.stream)