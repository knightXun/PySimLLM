import math
import sys
from enum import Enum

from MemBus import MemBus
from MyPacket import MyPacket
from Algorithm import Algorithm
from MockNcclLog import MockNcclLog, NcclLogLevel
from Sys import Sys
from StreamStat import StreamStat
from Common import ComType, EventType, StreamState

from system.topology.RingTopology import RingTopology
from AstraNetworkAPI import sim_request, req_type_e
from system.MockNcclChannel import *
from system.SendPacketEventHandlerData import SendPacketEventHandlerData
from PacketBundle import PacketBundle
from system.RecvPacketEventHandlerData import RecvPacketEventHandlerData

class HalvingDoubling(Algorithm):
    def __init__(self, type, id, layer_num, ring_topology, data_size, boost_mode):
        super().__init__(layer_num)
        self.comType = type
        self.id = id
        self.logicalTopology = ring_topology
        self.data_size = data_size
        self.nodes_in_ring = ring_topology.get_nodes_in_ring()
        self.parallel_reduce = 1
        self.total_packets_sent = 0
        self.total_packets_received = 0
        self.free_packets = 0
        self.zero_latency_packets = 0
        self.non_zero_latency_packets = 0
        self.toggle = False
        self.name = "HalvingDoubling"
        self.enabled = True
        if boost_mode:
            self.enabled = ring_topology.is_enabled()
        if ring_topology.dimension == RingTopology.Dimension.Local:
            self.transmition = "Fast"
        else:
            self.transmition = "Usual"
        if type == ComType.All_Reduce:
            self.stream_count = 2 * int(math.log2(self.nodes_in_ring))
        else:
            self.stream_count = int(math.log2(self.nodes_in_ring))
        if type == ComType.All_Gather:
            self.max_count = 0
        else:
            self.max_count = int(math.log2(self.nodes_in_ring))
        self.remained_packets_per_message = 1
        self.remained_packets_per_max_count = 1
        if type == ComType.All_Reduce:
            self.final_data_size = data_size
            self.msg_size = data_size // 2
            self.rank_offset = 1
            self.offset_multiplier = 2
        elif type == ComType.All_Gather:
            self.final_data_size = data_size * self.nodes_in_ring
            self.msg_size = data_size
            self.rank_offset = self.nodes_in_ring // 2
            self.offset_multiplier = 0.5
        elif type == ComType.Reduce_Scatter:
            self.final_data_size = data_size // self.nodes_in_ring
            self.msg_size = data_size // 2
            self.rank_offset = 1
            self.offset_multiplier = 2
        else:
            print("######### Exiting because of unknown communication type for HalveingDoubling collective algorithm #########")
            sys.exit(1)
        direction = self.specify_direction()
        self.current_receiver = id
        for i in range(self.rank_offset):
            self.current_receiver = ring_topology.get_receiver_node(self.current_receiver, direction)
            self.current_sender = self.current_receiver

    def get_non_zero_latency_packets(self):
        return int(math.log2(self.nodes_in_ring)) - 1 * self.parallel_reduce

    def specify_direction(self):
        if self.rank_offset == 0:
            return RingTopology.Direction.Clockwise
        reminder = (self.logicalTopology.index_in_ring // self.rank_offset) % 2
        if reminder == 0:
            return RingTopology.Direction.Clockwise
        else:
            return RingTopology.Direction.Anticlockwise

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
            self.rank_offset *= self.offset_multiplier
            self.msg_size //= self.offset_multiplier
            if self.rank_offset == self.nodes_in_ring and self.comType == ComType.All_Reduce:
                self.offset_multiplier = 0.5
                self.rank_offset *= self.offset_multiplier
                self.msg_size //= self.offset_multiplier
            self.current_receiver = self.id
            direction = self.specify_direction()
            for i in range(self.rank_offset):
                self.current_receiver = self.logicalTopology.get_receiver_node(self.current_receiver, direction)
                self.current_sender = self.current_receiver

    def reduce(self):
        self.process_stream_count()
        self.packets.pop(0)
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
            packet = MyPacket(self.msg_size, self.stream.current_queue_id, self.current_sender, self.current_receiver)
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
            packet = MyPacket(self.msg_size, self.stream.current_queue_id, self.current_sender, self.current_receiver)
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
        snd_req = {
            "srcRank": self.id,
            "dstRank": packet.preferred_dest,
            "tag": self.stream.stream_num,
            "reqType": req_type_e.UINT8,
            "vnet": self.stream.current_queue_id,
            "layerNum": self.layer_num
        }
        self.stream.owner.front_end_sim_send(
            0,
            Sys.dummy_data,
            packet.msg_size,
            req_type_e.UINT8,
            packet.preferred_dest,
            self.stream.stream_num,
            snd_req,
            Sys.handleEvent,
            None
        )
        rcv_req = {
            "vnet": self.stream.current_queue_id,
            "layerNum": self.layer_num
        }
        ehd = RecvPacketEventHandlerData(
            self.stream,
            self.stream.owner.id,
            EventType.PacketReceived,
            packet.preferred_vnet,
            packet.stream_num
        )
        self.stream.owner.front_end_sim_recv(
            0,
            Sys.dummy_data,
            packet.msg_size,
            req_type_e.UINT8,
            packet.preferred_src,
            self.stream.stream_num,
            rcv_req,
            Sys.handleEvent,
            ehd
        )
        self.reduce()
        return True

    def exit(self):
        if len(self.packets) != 0:
            self.packets = []
        if len(self.locked_packets) != 0:
            self.locked_packets = []
        self.stream.owner.proceed_to_next_vnet_baseline(self.stream)
        return