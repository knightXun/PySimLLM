import time

class Callable:
    pass

class CallData:
    pass

class SharedBusStat:
    def __init__(self, bus_type, a, b, c, d):
        self.bus_type = bus_type
        self.total_shared_bus_transfer_delay = 0
        self.total_shared_bus_processing_delay = 0
        self.total_shared_bus_processing_queue_delay = 0
        self.total_shared_bus_transfer_queue_delay = 0

class Sys:
    @staticmethod
    def boostedTick():
        return time.time()

class LogGP:
    talking_it = None

    def call(self, callEvent, data):
        pass

class MemMovRequest(Callable, SharedBusStat):
    id = 0

    def __init__(self, request_num, generator, loggp, size, latency, callable, processed, send_back):
        super().__init__("Mem", 0, 0, 0, 0)
        self.size = size
        self.latency = latency
        self.callable = callable
        self.processed = processed
        self.send_back = send_back
        self.my_id = MemMovRequest.id
        MemMovRequest.id += 1
        self.generator = generator
        self.loggp = loggp
        self.total_transfer_queue_time = 0
        self.total_transfer_time = 0
        self.total_processing_queue_time = 0
        self.total_processing_time = 0
        self.request_num = request_num
        self.start_time = Sys.boostedTick()
        self.mem_bus_finished = True
        self.callEvent = "General"
        self.pointer = None

    def call(self, event, data):
        self.update_bus_stats(data)
        self.total_mem_bus_transfer_delay += data.total_shared_bus_transfer_delay
        self.total_mem_bus_processing_delay += data.total_shared_bus_processing_delay
        self.total_mem_bus_processing_queue_delay += data.total_shared_bus_processing_queue_delay
        self.total_mem_bus_transfer_queue_delay += data.total_shared_bus_transfer_queue_delay
        self.mem_request_counter = 1
        self.mem_bus_finished = True
        self.loggp.talking_it = self.pointer
        self.loggp.call(self.callEvent, data)

    def update_bus_stats(self, data):
        # 假设这里的更新逻辑是简单的累加，可根据实际情况修改
        self.total_shared_bus_transfer_delay += data.total_shared_bus_transfer_delay
        self.total_shared_bus_processing_delay += data.total_shared_bus_processing_delay
        self.total_shared_bus_processing_queue_delay += data.total_shared_bus_processing_queue_delay
        self.total_shared_bus_transfer_queue_delay += data.total_shared_bus_transfer_queue_delay

    def wait_wait_for_mem_bus(self, pointer):
        self.mem_bus_finished = False
        self.pointer = pointer

    def set_iterator(self, pointer):
        self.pointer = pointer

    