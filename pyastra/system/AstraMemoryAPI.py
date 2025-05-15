import abc


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
