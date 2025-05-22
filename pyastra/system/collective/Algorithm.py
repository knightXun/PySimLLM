from enum import Enum

from BaseStream import BaseStream
from system.CallData import CallData
from system.Callable import Callable
from system.Common import ComType

class Algorithm(Callable):
    class Name(Enum):
        Ring = 0
        DoubleBinaryTree = 1
        AllToAll = 2
        HalvingDoubling = 3

    def __init__(self, layer_num):
        self.name = None
        self.id = None
        self.stream = None
        self.logicalTopology = None
        self.data_size = 0
        self.final_data_size = 0
        self.comType = None
        self.enabled = True
        self.layer_num = layer_num

    def __del__(self):
        pass

    def run(self, event, data):
        raise NotImplementedError

    def exit(self):
        if self.stream and hasattr(self.stream, 'owner') and hasattr(self.stream.owner, 'proceed_to_next_vnet_baseline'):
            self.stream.owner.proceed_to_next_vnet_baseline(self.stream)

    def init(self, stream):
        self.stream = stream

    def call(self, event, data):
        pass