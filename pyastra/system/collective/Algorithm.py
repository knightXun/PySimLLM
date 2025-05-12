from enum import Enum

# 假设这些类已经在其他地方定义
class BaseStream:
    pass

class CallData:
    pass

class EventType:
    pass

class Callable:
    pass

class ComType:
    pass

class LogicalTopology:
    pass


class Algorithm(Callable):
    class Name(Enum):
        Ring = 1
        DoubleBinaryTree = 2
        AllToAll = 3
        HalvingDoubling = 4

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