import time

from Callable import Callable
from Sys import Sys 
from LogGP import LogGP
from ShareBusStat import SharedBusStat
from CallData import CallData
from Common import EventType

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
        self.callEvent = EventType.General
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
        self.total_shared_bus_transfer_delay += data.total_shared_bus_transfer_delay
        self.total_shared_bus_processing_delay += data.total_shared_bus_processing_delay
        self.total_shared_bus_processing_queue_delay += data.total_shared_bus_processing_queue_delay
        self.total_shared_bus_transfer_queue_delay += data.total_shared_bus_transfer_queue_delay

    def wait_wait_for_mem_bus(self, pointer):
        self.mem_bus_finished = False
        self.pointer = pointer

    def set_iterator(self, pointer):
        self.pointer = pointer

    