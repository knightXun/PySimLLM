import time
import math
from collections import namedtuple

# 定义 Tick 类型，这里简单用 int 代替
Tick = int

# 定义 BusType 枚举
BusType = namedtuple('BusType', ['Shared'])
BusType.Shared = 'Shared'

# 定义 EventType 枚举
EventType = namedtuple('EventType', ['MA_to_NPU', 'NPU_to_MA'])
EventType.MA_to_NPU = 'MA_to_NPU'
EventType.NPU_to_MA = 'NPU_to_MA'

# 定义 SharedBusStat 类
class SharedBusStat:
    def __init__(self, bus_type, param1, param2, param3, param4):
        self.bus_type = bus_type
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4

# 假设 Callable 类
class Callable:
    def __call__(self):
        pass

# 假设 Sys 类
class Sys:
    def register_event(self, callable_obj, event_type, bus_stat, delay):
        # 这里简单打印事件信息，实际使用时可根据需求修改
        print(f"Registering event: {event_type} with delay {delay}")
        time.sleep(delay / 1000)  # 模拟延迟
        callable_obj()

# 假设 LogGP 类
class LogGP:
    def __init__(self, side, generator, L, o, g, G, event_type):
        self.side = side
        self.generator = generator
        self.L = L
        self.o = o
        self.g = g
        self.G = G
        self.event_type = event_type
        self.partner = None

    def request_read(self, bytes, processed, send_back, callable_obj):
        print(f"Requesting read on {self.side} side with {bytes} bytes")

    def attach_mem_bus(self, generator, L, o, g, param, model_shared_bus, communication_delay):
        print(f"Attaching memory bus on {self.side} side")

# 假设 MockNcclLog 类
class MockNcclLog:
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

class MemBus:
    class Transmition:
        Fast = 'Fast'
        Usual = 'Usual'

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
                NcclLog = MockNcclLog.getInstance()
                self.generator.register_event(callable_obj, EventType.NPU_to_MA, SharedBusStat(BusType.Shared, 0, 10, 0, 0), 10)
            else:
                self.generator.register_event(callable_obj, EventType.NPU_to_MA, SharedBusStat(BusType.Shared, 0, self.communication_delay, 0, 0), self.communication_delay)

    def send_from_MA_to_NPU(self, transmition, bytes, processed, send_back, callable_obj):
        NcclLog = MockNcclLog.getInstance()
        if self.model_shared_bus and transmition == self.Transmition.Usual:
            self.MA_side.request_read(bytes, processed, send_back, callable_obj)
        else:
            if transmition == self.Transmition.Fast:
                self.generator.register_event(callable_obj, EventType.MA_to_NPU, SharedBusStat(BusType.Shared, 0, 10, 0, 0), 10)
            else:
                self.generator.register_event(callable_obj, EventType.MA_to_NPU, SharedBusStat(BusType.Shared, 0, self.communication_delay, 0, 0), self.communication_delay)
    