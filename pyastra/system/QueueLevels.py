from enum import Enum

from AstraNetworkAPI import AstraNetworkAPI
from QueueLevelHandler import QueueLevelHandler
from topology.RingTopology import RingTopology

class QueueLevels:
    def __init__(self, levels_arg, queues_per_level=None, offset=0, backend: AstraNetworkAPI.BackendType=None):
        self.levels = []
        
        if isinstance(levels_arg, int) and queues_per_level is not None:
            total_levels = levels_arg
            start = offset
            for i in range(total_levels):
                tmp = QueueLevelHandler(i, start, start + queues_per_level - 1, backend)
                self.levels.append(tmp)
                start += queues_per_level
        
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

if __name__ == "__main__":
    ql1 = QueueLevels(total_levels=3, queues_per_level=5, offset=0, backend=AstraNetworkAPI.BackendType.MPI)
    
    lv = [2, 3, 4] 
    ql2 = QueueLevels(lv=lv, offset=10, backend=AstraNetworkAPI.BackendType.CUSTOM)