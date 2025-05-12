class AstraNetworkAPI:
    def sim_get_time(self):
        # 这里简单返回一个示例对象，实际使用时需要根据情况实现
        class Time:
            def __init__(self):
                self.time_val = 0
        return Time()


class AstraMemoryAPI:
    pass


class SimpleMemory(AstraMemoryAPI):
    def __init__(self, NI, access_latency, npu_access_bw_GB, nic_access_bw_GB):
        self.last_read_request_serviced = 0
        self.last_write_request_serviced = 0
        self.nic_read_request_count = 0
        self.nic_write_request_count = 0
        self.npu_read_request_count = 0
        self.npu_write_request_count = 0
        self.NI = NI
        self.access_latency = access_latency
        self.npu_access_bw_GB = npu_access_bw_GB
        self.nic_access_bw_GB = nic_access_bw_GB

    def set_network_api(self, astraNetworkApi):
        self.NI = astraNetworkApi

    def npu_mem_read(self, size):
        self.npu_read_request_count += 1
        delay = size / self.npu_access_bw_GB
        return delay

    def npu_mem_write(self, size):
        self.npu_write_request_count += 1
        delay = size / self.npu_access_bw_GB
        return delay

    def nic_mem_read(self, size):
        self.nic_read_request_count += 1
        time = self.NI.sim_get_time()
        time_ns = time.time_val
        delay = size / self.nic_access_bw_GB
        offset = 0
        if time_ns + self.access_latency < self.last_read_request_serviced:
            offset = (self.last_read_request_serviced + delay) - time_ns
            self.last_read_request_serviced += delay
        else:
            offset = (time_ns + self.access_latency + delay) - time_ns
            self.last_read_request_serviced = time_ns + self.access_latency + delay
        return int(offset)

    def nic_mem_write(self, size):
        self.nic_write_request_count += 1
        time = self.NI.sim_get_time()
        time_ns = time.time_val
        delay = size / self.nic_access_bw_GB
        offset = 0
        if time_ns + self.access_latency < self.last_write_request_serviced:
            offset = (self.last_write_request_serviced + delay) - time_ns
            self.last_write_request_serviced += delay
        else:
            offset = (time_ns + self.access_latency + delay) - time_ns
            self.last_write_request_serviced = time_ns + self.access_latency + delay
        return int(offset)

    def mem_read(self, size):
        return self.nic_mem_read(size)

    def mem_write(self, size):
        return self.nic_mem_write(size)

    