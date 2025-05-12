# 头文件部分的翻译
from typing import Union

# 假设 ComType 是一个自定义类型，这里简单用 int 表示，实际中需要根据情况修改
ComType = int

class Sys:
    pass

class Algorithm:
    def __init__(self):
        self.data_size = 0
        self.final_data_size = 0
        self.comType = ComType()
        self.enabled = True

    def init(self, stream):
        pass

class BaseStream:
    pass

class CollectivePhase:
    def __init__(self, generator: Union[Sys, None] = None, queue_id: int = -1, algorithm: Union[Algorithm, None] = None):
        self.generator = generator
        self.queue_id = queue_id
        self.algorithm = algorithm
        self.enabled = True
        if algorithm is not None:
            self.initial_data_size = algorithm.data_size
            self.final_data_size = algorithm.final_data_size
            self.comm_type = algorithm.comType
            self.enabled = algorithm.enabled
        else:
            self.initial_data_size = 0
            self.final_data_size = 0
            self.comm_type = ComType()

    def init(self, stream: BaseStream):
        if self.algorithm is not None:
            self.algorithm.init(stream)