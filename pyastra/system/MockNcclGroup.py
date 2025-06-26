from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Union
import weakref  
from collections import defaultdict
from collections import deque
from types import SimpleNamespace
from enum import Enum
import os
from enum import IntEnum

from Common import * 
from MockNccl import *
from MockNcclChannel import *

class GroupType(Enum):
    TP = 0
    DP = 1
    PP = 2
    EP = 3
    DP_EP = 4
    NONE = 5

@dataclass
class ncclInfo:
    coll: Any  
    tuneinfo: Any  
    algorithm: int = 0
    protocol: int = 0
    nChannels: int = 0
    nThreads: int = 0
    nBytes: int = 0 

@dataclass
class TuneInfo:
    nNodes: int
    nRanks: int
    nChannels: int
    collNetSupport: int
    nvlsSupport: int
    minCompCap: int
    maxCompCap: int
    graphs: List[ncclTopoGraph] = None  
    latencies: List[List[List[float]]] = None
    bandwidths: List[List[List[float]]] = None

    def __post_init__(self):
        if self.graphs is None:
            # graphs = std::vector<ncclTopoGraph*>(NCCL_NUM_ALGORITHMS, nullptr)
            self.graphs = [None] * NCCL_NUM_ALGORITHMS  

        if self.latencies is None:
            self.latencies = [
                [ [0.0]*NCCL_NUM_PROTOCOLS for _ in range(NCCL_NUM_ALGORITHMS) ] 
                for __ in range(NCCL_NUM_FUNCTIONS)
            ]

        if self.bandwidths is None:
            self.bandwidths = [
                [ [0.0]*NCCL_NUM_PROTOCOLS for _ in range(NCCL_NUM_ALGORITHMS) ] 
                for __ in range(NCCL_NUM_FUNCTIONS)
            ]


@dataclass
class GroupInfo:
  group_index: int
  type: GroupType
  nNodes: int
  nRanks: int
  Ranks: List[int]
  NVSwitchs: List[int]

  def __init__(self, _group_index, _type, _nNodes, _nRanks, _Ranks, _NVSwitchs):
    self.group_index = _group_index
    self.type = _type
    self.nRanks = _nRanks
    self.Ranks = _Ranks
    self.nNodes = _nNodes
    self.NVSwitchs = _NVSwitchs


class DoubleBinaryTreeNode:
  node: int 
  left: None
  right: None

  def __init__(self, node: int, left: None, right: None):
    self.node = node
    self.left = left
    self.right = right

class MockNcclGroup:

  GroupIndex: Dict[Tuple[int, GroupType], int] = {}
  AllGroups: Dict[int, GroupInfo] = {}
#   Allringchannels: Dict[int, RingChannels] = {}
#   AllNVLStreechannels: Dict[int, NVLStreechannels] = {}
#   Alltreechannels: Dict[int, TreeChannels] = {}
#   AllNVLSchannels: Dict[int, TreeChannels] = {}
    
  g_flow_id: int = 0
  gpu_type: GPUType = GPUType.NONE
  FlowName2nums: Dict[str, int] = {}
#   flow_models: Dict[str, Dict[int, FlowModels]] = {}
  nccl_infos: Dict[str, ncclInfo] = {}

  def __init__(self, _ngpus, _gpus_per_nodes, _TP_size, _DP_size, _PP_size, _EP_size, _DP_EP_size, _NVSwitch, _gpu_type):
    self.g_flow_id = 0
    self.gpu_type = _gpu_type
    self.GroupIndex = {}
    self.AllGroups = {}
    
    all_group_idx = 0
    nNodes = _ngpus // _gpus_per_nodes
    nlocalranks = _gpus_per_nodes
    TP_nums = _ngpus // _TP_size
    DP_nums = _ngpus // _DP_size
    PP_nums = _ngpus // _PP_size
    EP_nums = _ngpus // _EP_size
    DP_EP_nums = _ngpus // _DP_EP_size
    
    
    nNodesPerTPGroup = _TP_size // nlocalranks + (_TP_size % nlocalranks > 0)
    
    # init TP group
    if _TP_size > 1:
        for i in range(TP_nums):
            ranks = []
            TPnodes = set()
            for j in range(_TP_size):
                rank = i * _TP_size + j
                ranks.append(rank)
                self.GroupIndex[(rank, GroupType.TP)] = all_group_idx
                node_idx = rank // _gpus_per_nodes
                TPnodes.add(node_idx)
            
            NVSwitchs = [_NVSwitch[idx] for idx in TPnodes]
            for idx in TPnodes:
                self.GroupIndex[(_NVSwitch[idx], GroupType.TP)] = all_group_idx
            
            self.AllGroups[all_group_idx] = GroupInfo(
                all_group_idx, GroupType.TP, nNodesPerTPGroup, _TP_size, ranks, NVSwitchs)
            all_group_idx += 1
    
    # init DP group
    if _DP_size > 1:
        for i in range(DP_nums):
            ranks = []
            DPnodes = set()
            for j in range(_DP_size):
                rank = i + j * DP_nums
                ranks.append(rank)
                self.GroupIndex[(rank, GroupType.DP)] = all_group_idx
                node_idx = rank // _gpus_per_nodes
                DPnodes.add(node_idx)
            
            NVSwitchs = [_NVSwitch[idx] for idx in DPnodes]
            for idx in DPnodes:
                self.GroupIndex[(_NVSwitch[idx], GroupType.DP)] = all_group_idx
            
            self.AllGroups[all_group_idx] = GroupInfo(
                all_group_idx, GroupType.DP, len(DPnodes), _DP_size, ranks, NVSwitchs)
            all_group_idx += 1
    
    # init PP group
    if _PP_size > 1:
        pass  # 原代码未实现，保持一致
    
    # init EP
    AllTPGroups = {idx: group for idx, group in self.AllGroups.items() if group.type == GroupType.TP}
    
    if _EP_size > 1:
        for i in range(TP_nums // _EP_size):
            TP_idx = i * _EP_size
            for j in range(_EP_size):
                for k in range(len(AllTPGroups[TP_idx].Ranks)):
                    ranks = []
                    EPnodes = set()
                    for l in range(TP_idx, TP_idx + _EP_size):
                        tmp_rank = AllTPGroups[l].Ranks[k]
                        node_idx = tmp_rank // _gpus_per_nodes
                        ranks.append(tmp_rank)
                        self.GroupIndex[(tmp_rank, GroupType.EP)] = all_group_idx
                        EPnodes.add(node_idx)
                    
                    NVSwitchs = [_NVSwitch[idx] for idx in EPnodes]
                    for idx in EPnodes:
                        self.GroupIndex[(_NVSwitch[idx], GroupType.EP)] = all_group_idx
                    
                    self.AllGroups[all_group_idx] = GroupInfo(
                        all_group_idx, GroupType.EP, len(EPnodes), _EP_size, ranks, NVSwitchs)
                    all_group_idx += 1
    
    # init EP_DP
    if _DP_EP_size > 1:
        for i in range(TP_nums // _DP_EP_size):
            TP_idx = i
            for j in range(_DP_EP_size):
                for k in range(len(AllTPGroups[TP_idx].Ranks)):
                    ranks = []
                    DP_EP_nodes = set()
                    for l in range(TP_idx, TP_idx + _DP_EP_size * _EP_size, _EP_size):
                        tmp_rank = AllTPGroups[l].Ranks[k]
                        node_idx = tmp_rank // _gpus_per_nodes
                        ranks.append(tmp_rank)
                        self.GroupIndex[(tmp_rank, GroupType.DP_EP)] = all_group_idx
                        DP_EP_nodes.add(node_idx)
                    
                    NVSwitchs = [_NVSwitch[idx] for idx in DP_EP_nodes]
                    for idx in DP_EP_nodes:
                        self.GroupIndex[(_NVSwitch[idx], GroupType.DP_EP)] = all_group_idx
                    
                    self.AllGroups[all_group_idx] = GroupInfo(
                        all_group_idx, GroupType.DP_EP, len(DP_EP_nodes), _DP_EP_size, ranks, NVSwitchs)
                    all_group_idx += 1