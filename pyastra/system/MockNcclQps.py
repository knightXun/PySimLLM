from collections import deque
from typing import Dict, Tuple

# 定义类型别名（Python 3.10+ 支持TypeAlias声明，这里用注释说明）
# PeerWaitingTasks: 键类型为 (int, (int, int))，值类型为 deque[int]
# PeerQps: 键类型为 (int, (int, int))，值类型为 int

class NcclQps:
    def __init__(self):
        # 初始化对等QPS字典
        self.peer_qps: Dict[Tuple[int, Tuple[int, int]], int] = {}
        # 初始化对等等待任务队列（使用deque实现队列）
        self.peer_wating_tasks: Dict[Tuple[int, Tuple[int, int]], deque[int]] = {}

    def __del__(self):
        # Python的垃圾回收会自动处理资源释放，这里显式声明析构方法保持与C++对应
        pass