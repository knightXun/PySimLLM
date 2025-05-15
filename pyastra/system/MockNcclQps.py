from collections import deque
from typing import Dict, Tuple


class NcclQps:
    def __init__(self):
        self.peer_qps: Dict[Tuple[int, Tuple[int, int]], int] = {}
        self.peer_wating_tasks: Dict[Tuple[int, Tuple[int, int]], deque[int]] = {}

    def __del__(self):
        pass