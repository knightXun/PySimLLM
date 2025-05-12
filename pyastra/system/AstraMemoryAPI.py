import abc

class ComputeMetaData:
    def __init__(self):
        self.compute_delay = 0


class ComputeAPI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute(self, M, K, N, msg_handler, fun_arg):
        pass

    @abc.abstractmethod
    def __del__(self):
        pass


class AstraMemoryAPI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def mem_read(self, size):
        pass

    @abc.abstractmethod
    def mem_write(self, size):
        pass

    @abc.abstractmethod
    def npu_mem_read(self, size):
        pass

    @abc.abstractmethod
    def npu_mem_write(self, size):
        pass

    @abc.abstractmethod
    def nic_mem_read(self, size):
        pass

    @abc.abstractmethod
    def nic_mem_write(self, size):
        pass

    def __del__(self):
        pass
