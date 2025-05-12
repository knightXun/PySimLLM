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
