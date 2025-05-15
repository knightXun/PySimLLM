# 假设这些类已经在其他地方定义好了
from typing import List

from Callable import Callable
from StreamStat import StreamStat
from RecvPacketEventHandlerData import RecvPacketEventHandlerData
from SendPacketEventHandlerData import SendPacketEventHandlerData
from Common import ComType, SchedulingPolicy, StreamState
from DataSet import DataSet
from StreamStat import StreamStat
from CollectivePhase import CollectivePhase
from Sys import Sys


class BaseStream(Callable, StreamStat):
    def __init__(self, stream_num, owner, phases_to_go: List[CollectivePhase]):
        self.stream_num = stream_num
        self.owner = owner
        self.initialized = False
        self.phases_to_go = phases_to_go
        for vn in self.phases_to_go:
            if vn.algorithm is not None:
                vn.init(self)
        self.state = StreamState.Created
        self.preferred_scheduling = SchedulingPolicy.NONE
        self.creation_time = Sys.boostedTick()
        self.total_packets_sent = 0
        self.current_queue_id = -1
        self.priority = 0
        self.my_current_phase = None
        self.current_com_type = None
        self.last_init = None
        self.dataset = None
        self.steps_finished = 0
        self.initial_data_size = 0
        self.last_phase_change = None
        self.test = 0
        self.test2 = 0
        self.phase_latencies = [0] * 10

    def changeState(self, state):
        self.state = state

    def consume(self, message: RecvPacketEventHandlerData):
        raise NotImplementedError

    def sendcallback(self, messages: SendPacketEventHandlerData):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    