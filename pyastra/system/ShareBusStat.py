from CallData import CallData
from MemBus import BusType

class SharedBusStat(CallData):
    def __init__(self, bus_type, total_bus_transfer_queue_delay, 
                 total_bus_transfer_delay, total_bus_processing_queue_delay, 
                 total_bus_processing_delay):
        self.total_shared_bus_transfer_queue_delay = 0.0
        self.total_shared_bus_transfer_delay = 0.0
        self.total_shared_bus_processing_queue_delay = 0.0
        self.total_shared_bus_processing_delay = 0.0

        self.total_mem_bus_transfer_queue_delay = 0.0
        self.total_mem_bus_transfer_delay = 0.0
        self.total_mem_bus_processing_queue_delay = 0.0
        self.total_mem_bus_processing_delay = 0.0

        self.shared_request_counter = 0
        self.mem_request_counter = 0

        if bus_type == BusType.Shared:
            self.total_shared_bus_transfer_queue_delay = total_bus_transfer_queue_delay
            self.total_shared_bus_transfer_delay = total_bus_transfer_delay
            self.total_shared_bus_processing_queue_delay = total_bus_processing_queue_delay
            self.total_shared_bus_processing_delay = total_bus_processing_delay
        elif bus_type == BusType.Mem:
            self.total_mem_bus_transfer_queue_delay = total_bus_transfer_queue_delay
            self.total_mem_bus_transfer_delay = total_bus_transfer_delay
            self.total_mem_bus_processing_queue_delay = total_bus_processing_queue_delay
            self.total_mem_bus_processing_delay = total_bus_processing_delay

    def update_bus_stats(self, bus_type, shared_bus_stat):
        if bus_type == BusType.Shared:
            self.total_shared_bus_transfer_queue_delay += shared_bus_stat.total_shared_bus_transfer_queue_delay
            self.total_shared_bus_transfer_delay += shared_bus_stat.total_shared_bus_transfer_delay
            self.total_shared_bus_processing_queue_delay += shared_bus_stat.total_shared_bus_processing_queue_delay
            self.total_shared_bus_processing_delay += shared_bus_stat.total_shared_bus_processing_delay
            self.shared_request_counter += 1
        elif bus_type == BusType.Mem:
            self.total_mem_bus_transfer_queue_delay += shared_bus_stat.total_mem_bus_transfer_queue_delay
            self.total_mem_bus_transfer_delay += shared_bus_stat.total_mem_bus_transfer_delay
            self.total_mem_bus_processing_queue_delay += shared_bus_stat.total_mem_bus_processing_queue_delay
            self.total_mem_bus_processing_delay += shared_bus_stat.total_mem_bus_processing_delay
            self.mem_request_counter += 1
        else:
            self.total_shared_bus_transfer_queue_delay += shared_bus_stat.total_shared_bus_transfer_queue_delay
            self.total_shared_bus_transfer_delay += shared_bus_stat.total_shared_bus_transfer_delay
            self.total_shared_bus_processing_queue_delay += shared_bus_stat.total_shared_bus_processing_queue_delay
            self.total_shared_bus_processing_delay += shared_bus_stat.total_shared_bus_processing_delay
            
            self.total_mem_bus_transfer_queue_delay += shared_bus_stat.total_mem_bus_transfer_queue_delay
            self.total_mem_bus_transfer_delay += shared_bus_stat.total_mem_bus_transfer_delay
            self.total_mem_bus_processing_queue_delay += shared_bus_stat.total_mem_bus_processing_queue_delay
            self.total_mem_bus_processing_delay += shared_bus_stat.total_mem_bus_processing_delay
            
            self.shared_request_counter += 1
            self.mem_request_counter += 1

    def take_bus_stats_average(self):
        if self.shared_request_counter > 0:
            self.total_shared_bus_transfer_queue_delay /= self.shared_request_counter
            self.total_shared_bus_transfer_delay /= self.shared_request_counter
            self.total_shared_bus_processing_queue_delay /= self.shared_request_counter
            self.total_shared_bus_processing_delay /= self.shared_request_counter
        
        if self.mem_request_counter > 0:
            self.total_mem_bus_transfer_queue_delay /= self.mem_request_counter
            self.total_mem_bus_transfer_delay /= self.mem_request_counter
            self.total_mem_bus_processing_queue_delay /= self.mem_request_counter
            self.total_mem_bus_processing_delay /= self.mem_request_counter