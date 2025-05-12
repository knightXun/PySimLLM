from enum import Enum

# 对应C++中的AstraNetworkAPI::BackendType和RingTopology::Direction枚举
class BackendType(Enum):
    Garnet = 0  # 示例值，根据实际情况调整
    # 其他可能的BackendType成员

class Direction(Enum):
    Clockwise = 0
    Anticlockwise = 1

class QueueLevelHandler:
    def __init__(self, level: int, start: int, end: int, backend: BackendType):
        # 初始化队列列表（包含[start, end]闭区间内的所有整数）
        self.queues = list(range(start, end + 1))
        self.allocator = 0
        self.first_allocator = 0
        self.last_allocator = len(self.queues) // 2
        self.level = level
        self.backend = backend

    def get_next_queue_id(self) -> tuple[int, Direction]:
        # 判断方向逻辑
        if (self.backend != BackendType.Garnet or self.level > 0) and \
           len(self.queues) > 1 and self.allocator >= (len(self.queues) // 2):
            dir = Direction.Anticlockwise
        else:
            dir = Direction.Clockwise

        if not self.queues:
            return (-1, dir)
        
        # 获取并更新allocator（循环队列逻辑）
        tmp = self.queues[self.allocator]
        self.allocator += 1
        if self.allocator == len(self.queues):
            self.allocator = 0
        return (tmp, dir)

    def get_next_queue_id_first(self) -> tuple[int, Direction]:
        dir = Direction.Clockwise
        if not self.queues:
            return (-1, dir)
        
        # 获取并更新first_allocator（前半段循环队列）
        tmp = self.queues[self.first_allocator]
        self.first_allocator += 1
        if self.first_allocator == len(self.queues) // 2:
            self.first_allocator = 0
        return (tmp, dir)

    def get_next_queue_id_last(self) -> tuple[int, Direction]:
        dir = Direction.Anticlockwise
        if not self.queues:
            return (-1, dir)
        
        # 获取并更新last_allocator（后半段循环队列）
        tmp = self.queues[self.last_allocator]
        self.last_allocator += 1
        if self.last_allocator == len(self.queues):
            self.last_allocator = len(self.queues) // 2
        return (tmp, dir)