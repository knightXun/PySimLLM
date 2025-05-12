# 假设这些类已经在其他地方定义好了
from typing import List

class Callable:
    pass

class StreamStat:
    pass

class RecvPacketEventHadndlerData:
    pass

class SendPacketEventHandlerData:
    pass

class SchedulingPolicy:
    None = None

class ComType:
    pass

class Tick:
    pass

class DataSet:
    pass

class StreamState:
    Created = None

class CollectivePhase:
    def __init__(self):
        self.algorithm = None

    def init(self, stream):
        if self.algorithm is not None:
            # 这里需要根据实际的 algorithm 实现来调用相应的初始化方法
            pass

class Sys:
    @staticmethod
    def boostedTick():
        return 0


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
        self.preferred_scheduling = SchedulingPolicy.None
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

    def consume(self, message: RecvPacketEventHadndlerData):
        raise NotImplementedError

    def sendcallback(self, messages: SendPacketEventHandlerData):
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    