from typing import List, Dict, Union
from abc import ABC, abstractmethod
import math

# from MockNcclChannel import GroupType
# from MockNcclGroup import MockNcclGroup, ncclInfo


class SingleFlow:
    def __init__(self,
                 flow_id: int = 0,
                 src: int = 0,
                 dest: int = 0,
                 flow_size: int = 0,
                 prev: List[int] = [],
                 parent_flow_id: List[int] = [],
                 child_flow_id: List[int] = [],
                 channel_id: int = 0,
                 chunk_id: int = 0,
                 chunk_count: int = 0,
                 conn_type: str = ""):
        self.flow_id = flow_id
        self.src = src
        self.dest = dest
        self.flow_size = flow_size
        self.prev = prev
        self.parent_flow_id = parent_flow_id
        self.child_flow_id = child_flow_id
        self.channel_id = channel_id
        self.chunk_id = chunk_id
        self.chunk_count = chunk_count
        self.conn_type = conn_type

class State:
    Forward_Pass = 1
    Weight_Gradient = 2
    Input_Gradient = 3

class ComType:
    NONE = 0
    Reduce_Scatter = 1
    All_Gather = 2
    All_Reduce = 3
    All_to_All = 4
    All_Reduce_All_to_All = 5