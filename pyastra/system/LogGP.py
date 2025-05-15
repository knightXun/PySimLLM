import enum
from collections import deque

from Common import EventType
from Callable import Callable
from MemBus import MemBus
from MemMovRequest import MemMovRequest
from ShareBusStat import SharedBusStat
from Sys import Sys


class LogGP(Callable):
    class State(enum.Enum):
        Free = 1
        waiting = 2
        Sending = 3
        Receiving = 4

    class ProcState(enum.Enum):
        Free = 1
        Processing = 2

    def __init__(self, name, generator, L, o, g, G, trigger_event):
        self.request_num = 0
        self.name = name
        self.L = L
        self.o = o
        self.g = g
        self.G = G
        self.last_trans = 0
        self.curState = self.State.Free
        self.prevState = self.State.Free
        self.processing_state = self.ProcState.Free
        self.sends = deque()
        self.receives = deque()
        self.processing = deque()
        self.retirements = deque()
        self.pre_send = deque()
        self.pre_process = deque()
        self.talking_it = None
        self.partner = None
        self.generator = generator
        self.trigger_event = trigger_event
        self.subsequent_reads = 0
        self.THRESHOLD = 8
        self.local_reduction_delay = generator.local_reduction_delay
        self.NPU_MEM = None

    def __del__(self):
        if self.NPU_MEM is not None:
            del self.NPU_MEM

    def attach_mem_bus(self, generator, L, o, g, G, model_shared_bus, communication_delay):
        self.NPU_MEM = MemBus("NPU2", "MEM2", generator, L, o, g, G, model_shared_bus, communication_delay, False)

    def process_next_read(self):
        offset = 0
        if self.prevState == self.State.Sending:
            assert Sys.boostedTick() >= self.last_trans
            if (self.o + (Sys.boostedTick() - self.last_trans)) > self.g:
                offset = self.o
            else:
                offset = self.g - (Sys.boostedTick() - self.last_trans)
        else:
            offset = self.o
        tmp = self.sends.popleft()
        tmp.total_transfer_queue_time += Sys.boostedTick() - tmp.start_time
        self.partner.switch_to_receiver(tmp, offset)
        self.curState = self.State.Sending
        self.generator.register_event(self, EventType.Send_Finished, None, offset + (self.G * (tmp.size - 1)))

    def request_read(self, bytes, processed, send_back, callable):
        mr = MemMovRequest(self.request_num, self.generator, self, bytes, 0, callable, processed, send_back)
        self.request_num += 1
        if self.NPU_MEM is not None:
            mr.callEvent = EventType.Consider_Send_Back
            self.pre_send.append(mr)
            self.pre_send[-1].wait_wait_for_mem_bus(len(self.pre_send) - 1)
            self.NPU_MEM.send_from_MA_to_NPU(MemBus.Transmition.Usual, mr.size, False, False, self.pre_send[-1])
        else:
            self.sends.append(mr)
            if self.curState == self.State.Free:
                if self.subsequent_reads > self.THRESHOLD and self.partner.sends and self.partner.subsequent_reads <= self.THRESHOLD:
                    if self.partner.curState == self.State.Free:
                        self.partner.call(EventType.General, None)
                    return
                self.process_next_read()

    def switch_to_receiver(self, mr, offset):
        mr.start_time = Sys.boostedTick()
        self.receives.append(mr)
        self.prevState = self.curState
        self.curState = self.State.Receiving
        self.generator.register_event(self, EventType.Rec_Finished, None, offset + ((mr.size - 1) * self.G) + self.L + self.o)
        self.subsequent_reads = 0

    def call(self, event, data):
        if event == EventType.Send_Finished:
            self.last_trans = Sys.boostedTick()
            self.prevState = self.curState
            self.curState = self.State.Free
            self.subsequent_reads += 1
        elif event == EventType.Rec_Finished:
            assert self.receives
            self.receives[0].total_transfer_time += Sys.boostedTick() - self.receives[0].start_time
            self.receives[0].start_time = Sys.boostedTick()
            self.last_trans = Sys.boostedTick()
            self.prevState = self.curState
            if len(self.receives) < 2:
                self.curState = self.State.Free
            if self.receives[0].processed:
                if self.NPU_MEM is not None:
                    self.receives[0].processed = False
                    self.receives[0].loggp = self
                    self.receives[0].callEvent = EventType.Consider_Process
                    self.pre_process.append(self.receives.popleft())
                    self.pre_process[-1].wait_wait_for_mem_bus(len(self.pre_process) - 1)
                    self.NPU_MEM.send_from_NPU_to_MA(MemBus.Transmition.Usual, self.pre_process[-1].size, False, True, self.pre_process[-1])
                else:
                    self.receives[0].processed = False
                    self.processing.append(self.receives.popleft())
                if self.processing_state == self.ProcState.Free and self.processing:
                    self.processing[0].total_processing_queue_time += Sys.boostedTick() - self.processing[0].start_time
                    self.processing[0].start_time = Sys.boostedTick()
                    self.generator.register_event(self, EventType.Processing_Finished, None, ((self.processing[0].size // 100) * self.local_reduction_delay) + 50)
                    self.processing_state = self.ProcState.Processing
            elif self.receives[0].send_back:
                if self.NPU_MEM is not None:
                    self.receives[0].send_back = False
                    self.receives[0].callEvent = EventType.Consider_Send_Back
                    self.receives[0].loggp = self
                    self.pre_send.append(self.receives.popleft())
                    self.pre_send[-1].wait_wait_for_mem_bus(len(self.pre_send) - 1)
                    self.NPU_MEM.send_from_NPU_to_MA(MemBus.Transmition.Usual, self.pre_send[-1].size, False, True, self.pre_send[-1])
                else:
                    self.receives[0].send_back = False
                    self.sends.append(self.receives.popleft())
            else:
                if self.NPU_MEM is not None:
                    self.receives[0].callEvent = EventType.Consider_Retire
                    self.receives[0].loggp = self
                    self.retirements.append(self.receives.popleft())
                    self.retirements[-1].wait_wait_for_mem_bus(len(self.retirements) - 1)
                    self.NPU_MEM.send_from_NPU_to_MA(MemBus.Transmition.Usual, self.retirements[-1].size, False, False, self.retirements[-1])
                else:
                    tmp = SharedBusStat(BusType.Shared, self.receives[0].total_transfer_queue_time, self.receives[0].total_transfer_time, self.receives[0].total_processing_queue_time, self.receives[0].total_processing_time)
                    tmp.update_bus_stats(BusType.Mem, self.receives[0])
                    self.receives[0].callable.call(self.trigger_event, tmp)
                    self.receives.popleft()
        elif event == EventType.Processing_Finished:
            assert self.processing
            self.processing[0].total_processing_time += Sys.boostedTick() - self.processing[0].start_time
            self.processing[0].start_time = Sys.boostedTick()
            self.processing_state = self.ProcState.Free
            if self.processing[0].send_back:
                if self.NPU_MEM is not None:
                    self.processing[0].send_back = False
                    self.processing[0].loggp = self
                    self.processing[0].callEvent = EventType.Consider_Send_Back
                    self.pre_send.append(self.processing.popleft())
                    self.pre_send[-1].wait_wait_for_mem_bus(len(self.pre_send) - 1)
                    self.NPU_MEM.send_from_NPU_to_MA(MemBus.Transmition.Usual, self.pre_send[-1].size, False, True, self.pre_send[-1])
                else:
                    self.processing[0].send_back = False
                    self.sends.append(self.processing.popleft())
            else:
                if self.NPU_MEM is not None:
                    self.processing[0].callEvent = EventType.Consider_Retire
                    self.processing[0].loggp = self
                    self.retirements.append(self.processing.popleft())
                    self.retirements[-1].wait_wait_for_mem_bus(len(self.retirements) - 1)
                    self.NPU_MEM.send_from_NPU_to_MA(MemBus.Transmition.Usual, self.retirements[-1].size, False, False, self.retirements[-1])
                else:
                    tmp = SharedBusStat(BusType.Shared, self.processing[0].total_transfer_queue_time, self.processing[0].total_transfer_time, self.processing[0].total_processing_queue_time, self.processing[0].total_processing_time)
                    tmp.update_bus_stats(BusType.Mem, self.processing[0])
                    self.processing[0].callable.call(self.trigger_event, tmp)
                    self.processing.popleft()
            if self.processing:
                self.processing[0].total_processing_queue_time += Sys.boostedTick() - self.processing[0].start_time
                self.processing[0].start_time = Sys.boostedTick()
                self.processing_state = self.ProcState.Processing
                self.generator.register_event(self, EventType.Processing_Finished, None, ((self.processing[0].size // 100) * self.local_reduction_delay) + 50)
        elif event == EventType.Consider_Retire:
            tmp = SharedBusStat(BusType.Shared, self.retirements[0].total_transfer_queue_time, self.retirements[0].total_transfer_time, self.retirements[0].total_processing_queue_time, self.retirements[0].total_processing_time)
            movRequest = self.talking_it
            tmp.update_bus_stats(BusType.Mem, movRequest)
            movRequest.callable.call(self.trigger_event, tmp)
            self.retirements.remove(self.talking_it)
            del data
        elif event == EventType.Consider_Process:
            movRequest = self.talking_it
            self.processing.append(movRequest)
            self.pre_process.remove(self.talking_it)
            if self.processing_state == self.ProcState.Free and self.processing:
                self.processing[0].total_processing_queue_time += Sys.boostedTick() - self.processing[0].start_time
                self.processing[0].start_time = Sys.boostedTick()
                self.generator.register_event(self, EventType.Processing_Finished, None, ((self.processing[0].size // 100) * self.local_reduction_delay) + 50)
                self.processing_state = self.ProcState.Processing
            del data
        elif event == EventType.Consider_Send_Back:
            assert self.pre_send
            movRequest = self.talking_it
            self.sends.append(movRequest)
            self.pre_send.remove(self.talking_it)
            del data
        if self.curState == self.State.Free:
            if self.sends:
                if self.subsequent_reads > self.THRESHOLD and self.partner.sends and self.partner.subsequent_reads <= self.THRESHOLD:
                    if self.partner.curState == self.State.Free:
                        self.partner.call(EventType.General, None)
                    return
                self.process_next_read()

    