from enum import Enum

from AstraNetworkAPI import BackendType
from QueueLevelHandler import QueueLevelHandler
from topology.RingTopology import Direction


class QueueLevelHandler:
    def __init__(self, level: int, start: int, end: int, backend: BackendType):
        self.queues = list(range(start, end + 1))
        self.allocator = 0
        self.first_allocator = 0
        self.last_allocator = len(self.queues) // 2
        self.level = level
        self.backend = backend

    def get_next_queue_id(self) -> tuple[int, Direction]:
        if (self.backend != BackendType.Garnet or self.level > 0) and \
           len(self.queues) > 1 and self.allocator >= (len(self.queues) // 2):
            dir = Direction.Anticlockwise
        else:
            dir = Direction.Clockwise

        if len(self.queues) == 0:
            return (-1, dir)
        
        tmp = self.queues[self.allocator]
        self.allocator += 1
        if self.allocator == len(self.queues):
            self.allocator = 0
        return (tmp, dir)

    def get_next_queue_id_first(self) -> tuple[int, Direction]:
        dir = Direction.Clockwise
        if len(self.queues) == 0:
            return (-1, dir)
        
        tmp = self.queues[self.first_allocator]
        self.first_allocator += 1
        if self.first_allocator == len(self.queues) // 2:
            self.first_allocator = 0
        return (tmp, dir)

    def get_next_queue_id_last(self) -> tuple[int, Direction]:
        dir = Direction.Anticlockwise
        if len(self.queues) == 0:
            return (-1, dir)
        
        tmp = self.queues[self.last_allocator]
        self.last_allocator += 1
        if self.last_allocator == len(self.queues):
            self.last_allocator = len(self.queues) // 2
        return (tmp, dir)