# 定义一个类来模拟 Callable
class Callable:
    pass


class DMA_Request:
    def __init__(self, id, slots, latency, bytes, stream_owner=None):
        self.id = id
        self.slots = slots
        self.latency = latency
        self.executed = False
        self.bytes = bytes
        self.stream_owner = stream_owner
