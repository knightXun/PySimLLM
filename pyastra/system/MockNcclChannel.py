from typing import List, Dict, Union
from abc import ABC, abstractmethod
import math

from MockNcclChannel import GroupType
from MockNcclGroup import MockNcclGroup, ncclInfo


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

class ncclTree:
    def __init__(self, depth: int = 0, rank: int = 0, up: int = 0, down: List[int] = []):
        self.depth = depth
        self.rank = rank
        self.up = up
        self.down = down

class ncclChannelNode:
    def __init__(self, depth: int = 0, rank: int = 0, up = None, down: List = []):
        self.depth = depth
        self.rank = rank
        self.up = up
        self.down = down

class MockNcclComm:
    def __init__(self, rank: int, type: GroupType, GlobalGroup: MockNcclGroup):
        self.GlobalGroup = GlobalGroup
        self.type = type
        self.rank = rank
        self.ringchannels = self.GlobalGroup.genringchannels(rank, type)
        self.treechannels = self.GlobalGroup.gettreechannels(rank, type)
        self.nvlschannels = self.GlobalGroup.get_nvls_channels(rank, type)
        self.nvlstreechannels = None
        # self.nvlstreechannels = self.GlobalGroup.get_nvls_tree_channels(rank, type)

    def get_rings(self) -> Dict[int, Dict[int, List[int]]]:
        result = {}
        for ring_id, ring in self.ringchannels.items():
            for rank_key, value in ring.items():
                if rank_key not in result:
                    result[rank_key] = {}
                result[rank_key][ring_id] = value
        return result

    def get_treechannels(self) -> Dict[int, Dict[int, ncclTree]]:
        nvlschannel = {}
        nvlschannel[0] = {}
        for i in range(8):
            nvlschannel[0][i] = ncclTree(-1, i, 8, [])
        nvlschannel[0][8] = ncclTree(-1, 8, -1, [0, 1, 2, 3, 4, 5, 6, 7])
        return nvlschannel

    def get_nvls_channels(self) -> Dict[int, Dict[int, ncclTree]]:
        return self.nvlschannels

    def get_nvls_tree_channels(self):
        return self.nvlstreechannels

    def get_flow_model(self, data_size: int, collective_type: ComType, layer_num: int, loopstate: State):
        return self.GlobalGroup.getFlowModels(self.type, self.rank, collective_type, data_size, layer_num, loopstate)

    def get_algo_proto_info(self, data_size: int, collective_type: ComType) -> ncclInfo:
        return self.GlobalGroup.get_algo_proto_info(self.type, self.rank, collective_type, data_size)
    