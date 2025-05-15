from typing import Union
from Sys import Sys
from Common import ComType
from BaseStream import BaseStream
from collective.Algorithm import Algorithm

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