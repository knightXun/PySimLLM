import time
import math
from collections import namedtuple

from Common import EventType
from Callable import Callable
from Sys import Sys 
from LogGP import LogGP
from Common import SharedBusStat, BusType
from MockNcclLog import MockNcclLog

class MemBus:
    class Transmition:
        Fast = 0
        Usual = 1

    def __init__(self, side1, side2, generator, L, o, g, G, model_shared_bus, communication_delay, attach):
        self.NPU_side = LogGP(side1, generator, L, o, g, G, EventType.MA_to_NPU)
        self.MA_side = LogGP(side2, generator, L, o, g, G, EventType.NPU_to_MA)
        self.NPU_side.partner = self.MA_side
        self.MA_side.partner = self.NPU_side
        self.generator = generator
        self.model_shared_bus = model_shared_bus
        self.communication_delay = communication_delay
        if attach:
            self.NPU_side.attach_mem_bus(generator, L, o, g, 0.0038, model_shared_bus, communication_delay)

    def __del__(self):
        del self.NPU_side
        del self.MA_side

    def send_from_NPU_to_MA(self, transmition, bytes, processed, send_back, callable_obj):
        if self.model_shared_bus and transmition == self.Transmition.Usual:
            self.NPU_side.request_read(bytes, processed, send_back, callable_obj)
        else:
            if transmition == self.Transmition.Fast:
                NcclLog = MockNcclLog.get_instance()
                self.generator.register_event(callable_obj, EventType.NPU_to_MA, SharedBusStat(BusType.Shared, 0, 10, 0, 0), 10)
            else:
                self.generator.register_event(callable_obj, EventType.NPU_to_MA, SharedBusStat(BusType.Shared, 0, self.communication_delay, 0, 0), self.communication_delay)

    def send_from_MA_to_NPU(self, transmition, bytes, processed, send_back, callable_obj):
        NcclLog = MockNcclLog.get_instance()
        if self.model_shared_bus and transmition == self.Transmition.Usual:
            self.MA_side.request_read(bytes, processed, send_back, callable_obj)
        else:
            if transmition == self.Transmition.Fast:
                self.generator.register_event(callable_obj, EventType.MA_to_NPU, SharedBusStat(BusType.Shared, 0, 10, 0, 0), 10)
            else:
                self.generator.register_event(callable_obj, EventType.MA_to_NPU, SharedBusStat(BusType.Shared, 0, self.communication_delay, 0, 0), self.communication_delay)
    