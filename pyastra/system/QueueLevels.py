# -*- coding: utf-8 -*-
from enum import Enum

class AstraNetworkAPI:
    class BackendType(Enum):
        # 这里根据实际需求补充具体后端类型，示例仅作占位
        MPI = 1
        CUSTOM = 2

class RingTopology:
    class Direction(Enum):
        # 这里根据实际需求补充方向类型，示例仅作占位
        LEFT = 1
        RIGHT = 2

class QueueLevelHandler:
    # 假设这是原C++ QueueLevelHandler类的Python实现（需要根据实际情况补充完整）
    def __init__(self, level_id: int, start: int, end: int, backend: AstraNetworkAPI.BackendType):
        self.level_id = level_id
        self.start = start
        self.end = end
        self.backend = backend
        # 补充其他必要的成员变量和逻辑
    
    def get_next_queue_id(self) -> tuple[int, RingTopology.Direction]:
        # 补充实际逻辑
        return (0, RingTopology.Direction.LEFT)
    
    def get_next_queue_id_first(self) -> tuple[int, RingTopology.Direction]:
        # 补充实际逻辑
        return (0, RingTopology.Direction.LEFT)
    
    def get_next_queue_id_last(self) -> tuple[int, RingTopology.Direction]:
        # 补充实际逻辑
        return (0, RingTopology.Direction.LEFT)

class QueueLevels:
    def __init__(self, levels_arg, queues_per_level=None, offset=0, backend: AstraNetworkAPI.BackendType=None):
        """
        支持两种构造方式：
        1. QueueLevels(total_levels, queues_per_level, offset, backend)
        2. QueueLevels(lv: list[int], offset, backend)
        """
        self.levels = []
        
        # 处理第一种构造方式（层级数+每层级队列数）
        if isinstance(levels_arg, int) and queues_per_level is not None:
            total_levels = levels_arg
            start = offset
            for i in range(total_levels):
                tmp = QueueLevelHandler(i, start, start + queues_per_level - 1, backend)
                self.levels.append(tmp)
                start += queues_per_level
        
        # 处理第二种构造方式（层级队列数向量）
        elif isinstance(levels_arg, list) and all(isinstance(x, int) for x in levels_arg):
            lv = levels_arg
            start = offset
            level_id = 0
            for count in lv:
                tmp = QueueLevelHandler(level_id, start, start + count - 1, backend)
                self.levels.append(tmp)
                start += count
                level_id += 1
        
        else:
            raise ValueError("Invalid constructor arguments")

    def get_next_queue_at_level(self, level: int) -> tuple[int, RingTopology.Direction]:
        return self.levels[level].get_next_queue_id()

    def get_next_queue_at_level_first(self, level: int) -> tuple[int, RingTopology.Direction]:
        return self.levels[level].get_next_queue_id_first()

    def get_next_queue_at_level_last(self, level: int) -> tuple[int, RingTopology.Direction]:
        return self.levels[level].get_next_queue_id_last()

# 示例用法（需要根据实际情况调整）
if __name__ == "__main__":
    # 第一种构造方式示例
    ql1 = QueueLevels(total_levels=3, queues_per_level=5, offset=0, backend=AstraNetworkAPI.BackendType.MPI)
    
    # 第二种构造方式示例
    lv = [2, 3, 4]  # 三个层级，分别包含2/3/4个队列
    ql2 = QueueLevels(lv=lv, offset=10, backend=AstraNetworkAPI.BackendType.CUSTOM)