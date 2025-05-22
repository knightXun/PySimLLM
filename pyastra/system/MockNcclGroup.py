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
from MockNcclLog import NcclLogLevel
import MockNcclLog

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

FlowModels = Dict[Tuple[int, int], SingleFlow]
RingChannels = Dict[int, Dict[int, List[int]]]
NVLStreechannels = Dict[int, Dict[int, List[ncclChannelNode]]]
TreeChannels = Dict[int, Dict[int, ncclTree]]
TuneInfo_t = TuneInfo  # 或使用 'Optional[TuneInfo]' 表示可选值

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
  Allringchannels: Dict[int, RingChannels] = {}
  AllNVLStreechannels: Dict[int, NVLStreechannels] = {}
  Alltreechannels: Dict[int, TreeChannels] = {}
  AllNVLSchannels: Dict[int, TreeChannels] = {}
    
  g_flow_id: int = 0
  gpu_type: GPUType = GPUType.UNKNOWN
  FlowName2nums: Dict[str, int] = {}
  flow_models: Dict[str, Dict[int, FlowModels]] = {}
  nccl_infos: Dict[str, ncclInfo] = {}

  def __init__(self, _ngpus, _gpus_per_nodes, _TP_size, _DP_size, _PP_size, _EP_size, _DP_EP_size, _NVSwitch, _gpu_type):
    self.g_flow_id = 0
    self.gpu_type = _gpu_type
    self.GroupIndex = {}
    self.AllGroups = {}
    
    # init groups
    nccl_log = MockNcclLog.get_instance()
    if _ngpus % _gpus_per_nodes != 0 or _ngpus // _gpus_per_nodes <= 0:
        nccl_log.write_log(NcclLogLevel.ERROR, "The number of GPUs used is not a multiple of the number of GPUs per node.")
        return
    
    all_group_idx = 0
    nNodes = _ngpus // _gpus_per_nodes
    nlocalranks = _gpus_per_nodes
    TP_nums = _ngpus // _TP_size
    DP_nums = _ngpus // _DP_size
    PP_nums = _ngpus // _PP_size
    EP_nums = _ngpus // _EP_size
    DP_EP_nums = _ngpus // _DP_EP_size
    
    if (TP_nums <= 0 or DP_nums <= 0 or PP_nums <= 0 or EP_nums <= 0 or DP_EP_nums <= 0 or 
        (_TP_size * _DP_size * _PP_size != _ngpus) or (_EP_size * _DP_EP_size != _DP_size)):
        nccl_log.write_log(NcclLogLevel.ERROR, "The group division method is incorrect.")
        return
    
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

        
  def getFlowModels(self, type: GroupType, rank: int, op: ComType, 
                     data_size: int, layer_num: int, loopstate: State) -> FlowModels:

    flow_model_name = ""
    gp_info = None
    gp_idx = 0
    end_rank = 0
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.")
        return None
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    
    if type == GroupType.TP:
        flow_model_name = "TP"
    elif type == GroupType.DP:
        flow_model_name = "DP"
    elif type == GroupType.EP:
        flow_model_name = "EP"
    elif type == GroupType.DP_EP:
        flow_model_name = "DP_EP"
    
    flow_model_name = f"{flow_model_name}_{gp_idx}_{layer_num}_{loopstate.value}_{op.value}_{data_size}"
    
    if flow_model_name in self.flow_models:
        self.FlowName2nums[flow_model_name] += 1
        return self.flow_models[flow_model_name].get(rank, None)
    else:
        self.flow_models[flow_model_name] = self.genFlowModels(type, rank, op, data_size)
        self.FlowName2nums[flow_model_name] = 1
        return self.flow_models[flow_model_name][rank]
        
  def genFlowModels(self, type: GroupType, rank: int, op: ComType, 
                     data_size: int) -> Dict[int, FlowModels]:    
    if op == ComType.All_Reduce:
        return self.genAllReduceFlowModels(type, rank, data_size)
    elif op == ComType.All_Gather:
        return self.genAllGatherFlowModels(type, rank, data_size)
    elif op == ComType.Reduce_Scatter:
        return self.genReduceScatterFlowModels(type, rank, data_size)
    elif op == ComType.All_to_All:
        return self.genAlltoAllFlowModels(type, rank, data_size)
    
    return {}
        
  def genReduceScatterFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:    
    result = {}
    rank2flowmodels = defaultdict(dict)
    rank2pflowmodels = {}
    task_list = {}
    task_list2 = {}
    chunksize = 0
    send_size = 0
    nranks = 0
    chunkcount = 0
    chunkid = 0
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    ringchannels = self.Allringchannels[gp_idx]
    gp_info = self.AllGroups[gp_idx]
    
    PXN_ENABLE = os.getenv("AS_PXN_ENABLE") == "1"
    
    nranks = gp_info.nRanks
    chunkcount = nranks - 1
    chunksize = data_size // nranks // len(ringchannels)
    data_size = data_size // nranks // len(ringchannels)
    
    for ring_id, ring in ringchannels.items():
        task_list = {}
        send_size = 0
        chunkid = 0
        
        while send_size < data_size:
            real_chunksize = min(chunksize, data_size - send_size)
            prenoderecvrank = next(reversed(ring.values()))[2]
            prenodesendrank = next(reversed(ring.values()))[3]
            curnoderecvrank = next(iter(ring.values()))[2]
            curnodesendrank = next(iter(ring.values()))[3]
            prevranks = []
            
            for cur_rank, info in ring.items():
                if curnoderecvrank != info[2] and curnodesendrank != info[3]:
                    prenoderecvrank = curnoderecvrank
                    prenodesendrank = curnodesendrank
                    curnoderecvrank = info[2]
                    curnodesendrank = info[3]
                
                if info[3] == cur_rank and info[2] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                    prevranks = []
                    if info[0] != -1:
                        prevranks = [info[0]]
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        cur_rank,
                        info[2],
                        data_size,
                        prevranks,
                        [],
                        [self.g_flow_id + 1],
                        ring_id,
                        chunkid,
                        chunkcount,
                        "RING"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    self.g_flow_id += 1
                    
                    if cur_rank != -1:
                        prevranks = [cur_rank]
                    else:
                        prevranks = []
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        info[2],
                        info[1],
                        data_size,
                        prevranks,
                        [self.g_flow_id - 1],
                        [],
                        ring_id,
                        chunkid,
                        chunkcount,
                        "PXN_INIT"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    task_list[cur_rank] = tmp_result
                    self.g_flow_id += 1
                
                elif info[2] == cur_rank and info[3] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                    prevranks = []
                    if prenoderecvrank != -1:
                        prevranks = [prenoderecvrank]
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        cur_rank,
                        info[1],
                        data_size,
                        prevranks,
                        [],
                        [],
                        ring_id,
                        chunkid,
                        chunkcount,
                        "RING"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    task_list[cur_rank] = tmp_result
                    self.g_flow_id += 1
                
                else:
                    prevranks = []
                    if info[0] != -1:
                        prevranks = [info[0]]
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        cur_rank,
                        info[1],
                        data_size,
                        prevranks,
                        [],
                        [],
                        ring_id,
                        chunkid,
                        chunkcount,
                        "RING"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    task_list[cur_rank] = tmp_result
                    self.g_flow_id += 1
            
            chunkid += 1
            
            for _ in range(nranks - 2):
                task_list2 = {}
                prenoderecvrank = next(reversed(ring.values()))[2]
                prenodesendrank = next(reversed(ring.values()))[3]
                curnoderecvrank = next(iter(ring.values()))[2]
                curnodesendrank = next(iter(ring.values()))[3]
                
                for cur_rank, info in ring.items():
                    if curnoderecvrank != info[2] and curnodesendrank != info[3]:
                        prenoderecvrank = curnoderecvrank
                        prenodesendrank = curnodesendrank
                        curnoderecvrank = info[2]
                        curnodesendrank = info[3]
                    
                    partner_flow_id = task_list[info[0]].flow_id
                    
                    if info[3] == cur_rank and info[2] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                        prevranks = []
                        if info[0] != -1:
                            prevranks = [info[0]]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[2],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [self.g_flow_id + 1],
                            ring_id,
                            chunkid,
                            chunkcount,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                        
                        if cur_rank != -1:
                            prevranks = [cur_rank]
                        else:
                            prevranks = []
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            info[2],
                            info[1],
                            data_size,
                            prevranks,
                            [self.g_flow_id - 1],
                            [],
                            ring_id,
                            chunkid,
                            chunkcount,
                            "RING"
                        )
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                    
                    elif info[2] == cur_rank and info[3] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                        prevranks = []
                        if prenoderecvrank != -1:
                            prevranks = [prenoderecvrank]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[1],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [],
                            ring_id,
                            chunkid,
                            chunkcount,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                    
                    else:
                        prevranks = []
                        if info[0] != -1:
                            prevranks = [info[0]]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[1],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [],
                            ring_id,
                            chunkid,
                            chunkcount,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                
                task_list = task_list2
                chunkid += 1
            
            send_size += real_chunksize
    
    for key, flow in result.items():
        src = flow.src
        dst = flow.dest
        rank2flowmodels[src][(key[0], key[1])] = flow
        rank2flowmodels[dst][(key[0], key[1])] = flow
    
    for src, flows in rank2flowmodels.items():
        rank2pflowmodels[src] = FlowModels(flows)
    
    return rank2pflowmodels
        
  def genAlltoAllFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:    
    result = {}
    rank2flowmodels = defaultdict(dict)
    rank2pflowmodels = {}
    chunksize = 0
    send_size = 0
    nranks = 0
    chunkcount = 0
    chunkid = 0
    NcclLog = MockNcclLog.get_instance()
    ringchannels = None
    gp_idx = None

    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.")
        return {}
    else:
      gp_idx = self.GroupIndex[(rank, type)]
      ringchannels = self.Allringchannels[gp_idx]
      gp_info = self.AllGroups[gp_idx]
      
    nranks = gp_info.nRanks
    chunkcount = nranks - 1
    chunksize = data_size // nranks
    data_size = data_size // nranks
    
    for i, src_rank in enumerate(gp_info.Ranks):
        prev = [r for j, r in enumerate(gp_info.Ranks) if i != j]
        
        for j, dst_rank in enumerate(gp_info.Ranks):
            if i == j:
                continue
                
            tmp_result = SingleFlow(
                self.g_flow_id,
                src_rank,
                dst_rank,
                chunksize,
                prev,
                [],
                [],
                0,
                0,
                1,
                "RING"
            )
            result[(0, self.g_flow_id)] = tmp_result
            self.g_flow_id += 1
    
    for key, flow in result.items():
        src = flow.src
        dst = flow.dest
        rank2flowmodels[src][(key[0], key[1])] = flow
        rank2flowmodels[dst][(key[0], key[1])] = flow
    
    for src, flows in rank2flowmodels.items():
        rank2pflowmodels[src] = FlowModels(flows)
    
    return rank2pflowmodels
        
  def genAllReduceFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:
    ncc_info = self.get_algo_proto_info(type, rank, ComType.All_Reduce, data_size)
    
    if ncc_info.algorithm in (NCCL_ALGO_TREE, NCCL_ALGO_RING):
        return self.genAllReduceRingFlowModels(type, rank, data_size)
    elif ncc_info.algorithm == NCCL_ALGO_NVLS:
        return self.genAllreduceNVLSFlowModels(type, rank, data_size)
    elif ncc_info.algorithm == NCCL_ALGO_NVLS_TREE:
        return {}
    else:
        return {}
        
  def genAllReduceRingFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:    
    result = {}
    rank2flowmodels = defaultdict(dict)
    rank2pflowmodels = {}
    task_list = {}
    task_list2 = {}
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info and group ring channel, resulting in an error in generating the flow model.")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    ringchannels = self.Allringchannels[gp_idx]
    gp_info = self.AllGroups[gp_idx]
    nranks = gp_info.nRanks
    
    PXN_ENABLE = os.getenv("AS_PXN_ENABLE") == "1"
    
    chunksize = data_size // nranks // len(ringchannels)
    data_size = data_size // nranks // len(ringchannels)
    chunkcout = 2 * (gp_info.nRanks - 1)
    
    for ring_id, ring in ringchannels.items():
        task_list = {}
        send_size = 0
        chunk_id = 0
        
        while send_size < data_size:
            real_chunksize = min(chunksize, data_size - send_size)
            prenoderecvrank = next(reversed(ring.values()))[2]
            prenodesendrank = next(reversed(ring.values()))[3]
            curnoderecvrank = next(iter(ring.values()))[2]
            curnodesendrank = next(iter(ring.values()))[3]
            prevranks = []
            
            for cur_rank, info in ring.items():
                if curnoderecvrank != info[2] and curnodesendrank != info[3]:
                    prenoderecvrank = curnoderecvrank
                    prenodesendrank = curnodesendrank
                    curnoderecvrank = info[2]
                    curnodesendrank = info[3]
                
                if info[3] == cur_rank and info[2] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                    prevranks = []
                    if info[0] != -1:
                        prevranks = [info[0]]
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        cur_rank,
                        info[2],
                        data_size,
                        prevranks,
                        [],
                        [self.g_flow_id + 1],
                        ring_id,
                        chunk_id,
                        chunkcout,
                        "RING"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    self.g_flow_id += 1
                    
                    prevranks = [cur_rank]
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        info[2],
                        info[1],
                        data_size,
                        prevranks,
                        [self.g_flow_id - 1],
                        [],
                        ring_id,
                        chunk_id,
                        chunkcout,
                        "PXN_INIT"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    task_list[cur_rank] = tmp_result
                    self.g_flow_id += 1
                
                elif info[2] == cur_rank and info[3] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                    prevranks = []
                    if prenoderecvrank != -1:
                        prevranks = [prenoderecvrank]
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        cur_rank,
                        info[1],
                        data_size,
                        prevranks,
                        [],
                        [],
                        ring_id,
                        chunk_id,
                        chunkcout,
                        "RING"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    task_list[cur_rank] = tmp_result
                    self.g_flow_id += 1
                
                else:
                    prevranks = []
                    if info[0] != -1:
                        prevranks = [info[0]]
                    
                    tmp_result = SingleFlow(
                        self.g_flow_id,
                        cur_rank,
                        info[1],
                        data_size,
                        prevranks,
                        [],
                        [],
                        ring_id,
                        chunk_id,
                        chunkcout,
                        "RING"
                    )
                    result[(ring_id, self.g_flow_id)] = tmp_result
                    task_list[cur_rank] = tmp_result
                    self.g_flow_id += 1
            
            chunk_id += 1
            
            for _ in range(nranks - 1):
                task_list2 = {}
                prenoderecvrank = next(reversed(ring.values()))[2]
                prenodesendrank = next(reversed(ring.values()))[3]
                curnoderecvrank = next(iter(ring.values()))[2]
                curnodesendrank = next(iter(ring.values()))[3]
                
                for cur_rank, info in ring.items():
                    if curnoderecvrank != info[2] and curnodesendrank != info[3]:
                        prenoderecvrank = curnoderecvrank
                        prenodesendrank = curnodesendrank
                        curnoderecvrank = info[2]
                        curnodesendrank = info[3]
                    
                    partner_flow_id = task_list[info[0]].flow_id
                    
                    if info[3] == cur_rank and info[2] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                        prevranks = []
                        if info[0] != -1:
                            prevranks = [info[0]]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[2],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [self.g_flow_id + 1],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                        
                        prevranks = [cur_rank]
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            info[2],
                            info[1],
                            data_size,
                            prevranks,
                            [self.g_flow_id - 1],
                            [],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "PXN_INIT"
                        )
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                    
                    elif info[2] == cur_rank and info[3] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                        prevranks = []
                        if prenoderecvrank != -1:
                            prevranks = [prenoderecvrank]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[1],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                    
                    else:
                        prevranks = []
                        if info[0] != -1:
                            prevranks = [info[0]]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[1],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                
                task_list = task_list2
                chunk_id += 1
            
            for _ in range(nranks - 2):
                task_list2 = {}
                prenoderecvrank = next(reversed(ring.values()))[2]
                prenodesendrank = next(reversed(ring.values()))[3]
                curnoderecvrank = next(iter(ring.values()))[2]
                curnodesendrank = next(iter(ring.values()))[3]
                
                for cur_rank, info in ring.items():
                    if curnoderecvrank != info[2] and curnodesendrank != info[3]:
                        prenoderecvrank = curnoderecvrank
                        prenodesendrank = curnodesendrank
                        curnoderecvrank = info[2]
                        curnodesendrank = info[3]
                    
                    partner_flow_id = task_list[info[0]].flow_id
                    
                    if info[3] == cur_rank and info[2] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                        prevranks = []
                        if info[0] != -1:
                            prevranks = [info[0]]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[2],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [self.g_flow_id + 1],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                        
                        prevranks = []
                        if cur_rank != -1:
                            prevranks = [cur_rank]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            info[2],
                            info[1],
                            data_size,
                            prevranks,
                            [self.g_flow_id - 1],
                            [],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "PXN_INIT"
                        )
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                    
                    elif info[2] == cur_rank and info[3] != cur_rank and gp_info.nNodes > 1 and PXN_ENABLE:
                        prevranks = []
                        if prenoderecvrank != -1:
                            prevranks = [prenoderecvrank]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[1],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                    
                    else:
                        prevranks = []
                        if info[0] != -1:
                            prevranks = [info[0]]
                        
                        tmp_result = SingleFlow(
                            self.g_flow_id,
                            cur_rank,
                            info[1],
                            data_size,
                            prevranks,
                            [partner_flow_id],
                            [],
                            ring_id,
                            chunk_id,
                            chunkcout,
                            "RING"
                        )
                        result[(ring_id, partner_flow_id)].child_flow_id.append(self.g_flow_id)
                        task_list2[cur_rank] = tmp_result
                        result[(ring_id, self.g_flow_id)] = tmp_result
                        self.g_flow_id += 1
                
                task_list = task_list2
                chunk_id += 1
            
            send_size += real_chunksize
    
    for key, flow in result.items():
        src = flow.src
        dst = flow.dest
        rank2flowmodels[src][(key[0], key[1])] = flow
        rank2flowmodels[dst][(key[0], key[1])] = flow
    
    for src, flows in rank2flowmodels.items():
        rank2pflowmodels[src] = FlowModels(flows)
    
    return rank2pflowmodels
        
  def genAllreduceNVLSFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:    
    gp_info = GroupInfo()
    gp_idx = 0
    chunk_count = 4
    rank2flowmodels = defaultdict(dict)
    rank2pflowmodels = {}
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info , resulting in an error in genAllreduceNVLSFlowModels.")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    result = {}
    
    if gp_info.nNodes == 1:
        NVswitchs = gp_info.NVSwitchs
        ranks = gp_info.Ranks
        chunk_size = data_size // chunk_count
        
        for ck in range(chunk_count):
            for j in range(len(NVswitchs)):
                prevs = []
                parents = []
                
                for k in range(len(ranks)):
                    treeflow = SingleFlow(
                        self.g_flow_id,
                        ranks[k],
                        NVswitchs[j],
                        chunk_size,
                        [NVswitchs[j]],
                        [],
                        [],
                        0,
                        ck,
                        chunk_count,
                        "NVLS"
                    )
                    result[(0, self.g_flow_id)] = treeflow
                    prevs.append(ranks[k])
                    parents.append(self.g_flow_id)
                    self.g_flow_id += 1
                
                for k in range(len(ranks)):
                    treeflow = SingleFlow(
                        self.g_flow_id,
                        NVswitchs[j],
                        ranks[k],
                        chunk_size,
                        prevs,
                        parents,
                        [],
                        0,
                        ck,
                        chunk_count,
                        "NVLS"
                    )
                    result[(0, self.g_flow_id)] = treeflow
                    
                    for parent in parents:
                        result[(0, parent)].child_flow_id.append(self.g_flow_id)
                    
                    self.g_flow_id += 1
    
    for key, flow in result.items():
        src = flow.src
        dst = flow.dest
        rank2flowmodels[src][(key[0], key[1])] = flow
        rank2flowmodels[dst][(key[0], key[1])] = flow
    
    for src, flows in rank2flowmodels.items():
        rank2pflowmodels[src] = FlowModels(flows)
    
    return rank2pflowmodels
        
  def genallReduceNVLSTreeFlowModels(self, type: GroupType, rank: int, data_size: int) -> FlowModels:
    NcclLog = MockNcclLog.get_instance()
    gp_info = GroupInfo()
    gp_idx = 0
    chunk_count = 1
    result = {}
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no relevant group info, resulting in an error in generating genallReduceNVLSTreeFlowModels.")
        return None
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    nvlstreechannels = self.AllNVLStreechannels[gp_idx]
    NcclLog.write_log(NcclLogLevel.DEBUG, f" nvlstreechannels.size()  {len(nvlstreechannels)}")
    
    chunk_size = data_size // len(nvlstreechannels) // chunk_count
    
    for tree_id, tree_nodes_map in nvlstreechannels.items():
        if rank == 0:
            for node_rank, nodes_list in tree_nodes_map.items():
                NcclLog.write_log(NcclLogLevel.DEBUG, f" rank  {node_rank} nvls tree nodes ")
                for i, node in enumerate(nodes_list):
                    NcclLog.write_log(NcclLogLevel.DEBUG, f" node  {i} rank  {node.rank}")
                    if node.up is not None:
                        NcclLog.write_log(NcclLogLevel.DEBUG, f" up  {node.up.rank}")
                    NcclLog.write_log(NcclLogLevel.DEBUG, " down ")
                    for down_node in node.down:
                        NcclLog.write_log(NcclLogLevel.DEBUG, f"{down_node.rank} ")
        
        upinDegree = {}
        downinDegree = {}
        nodeprevs = {}
        
        for ck in range(chunk_count):
            nodeprevs = {}
            ncclchannelnodes = []
            
            for nodes_list in tree_nodes_map.values():
                for node in nodes_list:
                    ncclchannelnodes.append(node)
                    upinDegree[node] = len(node.down)
                    downinDegree[node] = 0 if node.up is None else 1
            
            self.generate_flow_model_nvls_tree_allreduce_up(
                ncclchannelnodes,
                upinDegree,
                nodeprevs,
                chunk_size,
                ck,
                chunk_count,
                tree_id,
                result
            )
            
            self.generate_flow_model_nvls_tree_allreduce_down(
                ncclchannelnodes,
                downinDegree,
                nodeprevs,
                chunk_size,
                ck,
                chunk_count,
                tree_id,
                result
            )
    
    return FlowModels(result)
        
  def generate_flow_model_nvls_tree_allreduce_up(self, nvlstreenodes: List[ncclChannelNode], 
                                                  upinDegree: Dict[ncclChannelNode, int], 
                                                  nodeprevs: Dict[ncclChannelNode, List[int]], 
                                                  chunk_size: int, chunk_id: int, chunk_count: int, 
                                                  channle_id: int, result: FlowModels) -> FlowModels:    
    q = deque()
    conn_tag = "NVLS_TREE"
    
    for node, degree in upinDegree.items():
        if degree == 0:
            q.append(node)
            nodeprevs[node] = []
    
    while q:
        current = q.popleft()
        
        if current.up is not None:
            upinDegree[current.up] -= 1
            
            if not current.down:
                _prev = [current.up.rank]
            else:
                _prev = [down.rank for down in current.down]
            
            tmp_result = SingleFlow(
                self.g_flow_id,
                current.rank,
                current.up.rank,
                chunk_size,
                _prev,
                nodeprevs[current],
                [],
                channle_id,
                chunk_id,
                chunk_count,
                conn_tag
            )
            
            for parent_flow_id in nodeprevs[current]:
                result[(channle_id, parent_flow_id)].child_flow_id.append(self.g_flow_id)
            
            result[(channle_id, self.g_flow_id)] = tmp_result
            self.g_flow_id += 1
            
            nodeprevs[current.up].append(tmp_result.flow_id)
            nodeprevs.pop(current, None)
            
            if upinDegree[current.up] == 0:
                q.append(current.up)
    
    return result
        
  def generate_flow_model_nvls_tree_allreduce_down(self, nvlstreenodes: List[ncclChannelNode], 
                                                    downinDegree: Dict[ncclChannelNode, int], 
                                                    nodeprevs: Dict[ncclChannelNode, List[int]], 
                                                    chunk_size: int, chunk_id: int, chunk_count: int, 
                                                    channle_id: int, result: FlowModels) -> FlowModels:
    q = deque()
    conn_tag = "NVLS_TREE"
    
    for node, degree in downinDegree.items():
        if degree == 0:
            q.append(node)
    
    while q:
        current = q.popleft()
        
        if current.down:
            for down in current.down:
                downinDegree[down] -= 1
                
                if current.up is None:
                    _prev = [down1.rank for down1 in current.down]
                else:
                    _prev = [current.up.rank]
                
                tmp_result = SingleFlow(
                    self.g_flow_id,
                    current.rank,
                    down.rank,
                    chunk_size,
                    _prev,
                    nodeprevs[current],
                    [],
                    channle_id,
                    chunk_id,
                    chunk_count,
                    conn_tag
                )
                
                for parent_flow_id in nodeprevs[current]:
                    result[(channle_id, parent_flow_id)].child_flow_id.append(self.g_flow_id)
                
                result[(channle_id, self.g_flow_id)] = tmp_result
                self.g_flow_id += 1
                
                nodeprevs[down].append(tmp_result.flow_id)
                
                if downinDegree[down] == 0:
                    q.append(down)
    
    return result
        
  def genAllReduceTreeFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:
    chunk_count = 64
    result = {}
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex or self.Alltreechannels.get(self.GroupIndex[(rank, type)]) is None:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info , resulting in an error in genAllreduceNVLSFlowModels.")
        return None
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    treechannels = self.Alltreechannels[gp_idx]
    chunk_size = data_size // len(treechannels) // chunk_count
    
    for tree_id, tree_nodes in treechannels.items():
        
        upinDegree = defaultdict(int)
        downinDegree = defaultdict(int)
        nodeprevs = defaultdict(list)
            
        for ck in range(chunk_count):
            nodeprevs = {};
            for node_id, node in tree_nodes.items():
                upinDegree[node_id] = len(node.down)
                downinDegree[node_id] = 0 if node.up == -1 else 1
            
            self.generate_flow_model_tree_allreduce_up(
                tree_nodes, upinDegree, nodeprevs, chunk_size, ck, chunk_count, tree_id, result
            )
            
            self.generate_flow_model_tree_allreduce_down(
                tree_nodes, downinDegree, nodeprevs, chunk_size, ck, chunk_count, tree_id, result
            )
    
    return FlowModels(result)
        
  def generate_flow_model_tree_allreduce_up(self, nodes: Dict[int, ncclTree], 
                                             upinDegree: Dict[int, int], 
                                             nodeprevs: Dict[int, List[int]], 
                                             chunk_size: int, chunk_id: int, chunk_count: int, 
                                             channle_id: int, result: FlowModels) -> FlowModels:    
    q = deque()
    conn_tag = "TREE_INIT"
    
    for node_id, degree in upinDegree.items():
        if degree == 0:
            q.append(nodes[node_id])
            nodeprevs[node_id] = []
    
    while q:
        current = q.popleft()
        
        if current.up != -1:
            upinDegree[current.up] -= 1
            
            if not current.down:
                _prev = [current.up]
            else:
                _prev = current.down
            
            tmp_result = SingleFlow(
                self.g_flow_id,
                current.rank,
                current.up,
                chunk_size,
                _prev,
                nodeprevs[current.rank],
                [],
                channle_id,
                chunk_id,
                chunk_count,
                conn_tag
            )
            
            for parent_flow_id in nodeprevs[current.rank]:
                result[(channle_id, parent_flow_id)].child_flow_id.append(self.g_flow_id)
            
            result[(channle_id, self.g_flow_id)] = tmp_result
            self.g_flow_id += 1
            
            nodeprevs[current.up].append(tmp_result.flow_id)
            nodeprevs.pop(current.rank, None)
            
            if upinDegree[current.up] == 0:
                q.append(nodes[current.up])
    
    return result
        
  def generate_flow_model_tree_allreduce_down(self, nodes: Dict[int, ncclTree], 
                                               downinDegree: Dict[int, int], 
                                               nodeprevs: Dict[int, List[int]], 
                                               chunk_size: int, chunk_id: int, chunk_count: int, 
                                               channle_id: int, result: FlowModels) -> FlowModels:
    q = deque()
    conn_tag = "TREE_INIT"
    
    for node_id, degree in downinDegree.items():
        if degree == 0:
            q.append(nodes[node_id])
    
    while q:
        current = q.popleft()
        
        if current.down:
            for down in current.down:
                downinDegree[down] -= 1
                
                if current.up == -1:
                    _prev = current.down
                else:
                    _prev = [current.up]
                
                tmp_result = SingleFlow(
                    self.g_flow_id,
                    current.rank,
                    down,
                    chunk_size,
                    _prev,
                    nodeprevs[current.rank],
                    [],
                    channle_id,
                    chunk_id,
                    chunk_count,
                    conn_tag
                )
                
                for parent_flow_id in nodeprevs[current.rank]:
                    result[(channle_id, parent_flow_id)].child_flow_id.append(self.g_flow_id)
                
                result[(channle_id, self.g_flow_id)] = tmp_result
                self.g_flow_id += 1
                
                nodeprevs[down].append(tmp_result.flow_id)
                
                if downinDegree[down] == 0:
                    q.append(nodes[down])
    
    return result
        
  def genAllGatherFlowModels(self, type: GroupType, rank: int, data_size: int) -> Dict[int, FlowModels]:
    pass
        
  def genInterDouBinTree(self, gp_info: GroupInfo) -> List[DoubleBinaryTreeNode]:
    q = []
    tmp_q = []
    result = []
    nNodes = gp_info.nNodes
    nodes = list(range(nNodes))
    
    for i in nodes:
        q.append(DoubleBinaryTreeNode(i))
    
    while len(q) > 1:
        tmp_q = []
        i = 0
        
        while (i + 2) < len(q):
            node0 = q[i]
            node1 = q[i + 1]
            node2 = q[i + 2]
            
            node1.left = node0
            node1.right = node2
            
            tmp_q.append(node1)
            
            if (i + 3) < len(q):
                node3 = q[i + 3]
                tmp_q.append(node3)
            
            i += 4
        
        remaining = len(q) - i
        
        if remaining == 1:
            node0 = q[i]
            tmp_q.append(node0)
        elif remaining == 2:
            node0 = q[i]
            node1 = q[i + 1]
            
            node1.left = node0
            tmp_q.append(node1)
        
        q = tmp_q
    
    root1 = self.InterDouBinTreeShift(q[0], nodes)
    chunk_count = 1
    
    for _ in range(chunk_count):
        result.append(q[0])
        result.append(root1)
    
    return result
        
  def InterDouBinTreeShift(self, root: DoubleBinaryTreeNode, nodes: List[int]) -> DoubleBinaryTreeNode:    
    node2treenode = {}
    rank2index = {}
    
    for i, node in enumerate(nodes):
        node2treenode[node] = DoubleBinaryTreeNode(node)
        rank2index[node] = i
    
    q = deque([root])
    
    while q:
        current = q.popleft()
        node = current.node
        nodeshift = nodes[(rank2index[node] + 1) % len(nodes)]
        currentshift = node2treenode[nodeshift]
        
        if current.left is not None:
            leftnode = current.left.node
            leftnodeshift = nodes[(rank2index[leftnode] + 1) % len(nodes)]
            currentshift.left = node2treenode[leftnodeshift]
            q.append(current.left)
        
        if current.right is not None:
            rightnode = current.right.node
            rightnodeshift = nodes[(rank2index[rightnode] + 1) % len(nodes)]
            currentshift.right = node2treenode[rightnodeshift]
            q.append(current.right)
    
    return node2treenode[nodes[(rank2index[root.node] + 1) % len(nodes)]]
        
  def ConnInterIntraTree(self, root: DoubleBinaryTreeNode, node2ranks: Dict[int, List[int]], 
                         TreeChannel: Dict[int, ncclTree]) -> None:
    if root is None:
        return
    
    ranks = node2ranks[root.node]
    
    for i in range(len(ranks) - 1):
        current = self.treechannel[ranks[i]]
        down = self.treechannel[ranks[i + 1]]
        current.down.append(ranks[i + 1])
        down.up = ranks[i]
    
    if root.left is not None:
        current = treechannel[ranks[0]]
        downrank = node2ranks[root.left.node][0]
        down = treechannel[downrank]
        current.down.append(downrank)
        down.up = ranks[0]
        self.ConnInterIntraTree(root.left, node2ranks, treechannel)
    
    if root.right is not None:
        current = treechannel[ranks[0]]
        downrank = node2ranks[root.right.node][0]
        down = treechannel[downrank]
        current.down.append(downrank)
        down.up = ranks[0]
        self.ConnInterIntraTree(root.right, node2ranks, treechannel)
        
  def generateringchannels(self, localrings: Dict[int, List[int]], 
                            groupInfo: GroupInfo, 
                            ringchannels: Dict[int, Dict[int, List[int]]]) -> None:
    nNodes = groupInfo.nNodes
    nlocalRanks = groupInfo.nRanks // nNodes
    delta = groupInfo.Ranks[nlocalRanks] - groupInfo.Ranks[0] if nNodes > 1 else 0
    
    for ring_id, ring_ranks in localrings.items():
        prev = -1
        next_node = -1
        
        for i in range(nNodes):
            node_recv = ring_ranks[0] + i * delta
            node_send = ring_ranks[nlocalRanks-1] + i * delta
            
            for j in range(nlocalRanks):
                current = ring_ranks[j] + i * delta
                
                if j == nlocalRanks-1:
                    next_node = ring_ranks[0] + (i + 1) * delta
                else:
                    next_node = ring_ranks[j+1] + i * delta
                    
                ringchannels.setdefault(ring_id, {})[current] = [prev, next_node, node_recv, node_send]
                prev = current
                
        end_rank = ring_ranks[nlocalRanks-1] + (nNodes - 1) * delta
        ringchannels[ring_id][ring_ranks[0]][0] = end_rank
        ringchannels[ring_id][end_rank][1] = ring_ranks[0]
        
  def gen_local_ring(self, rank: int, type: GroupType) -> Dict[int, List[int]]:    
    gp_info = GroupInfo()
    gp_idx = 0
    ranks = []
    localranks = []
    localrings = defaultdict(list)
    nNodes = 0
    nlocalranks = 0
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no relevant group info, resulting in an error in gen_local_ring")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    ranks = gp_info.Ranks
    nNodes = gp_info.nNodes
    nlocalranks = len(ranks) // nNodes
    ranks.sort()
    
    for i in range(nlocalranks):
        localranks.append(ranks[i])
        
    for i in range(nlocalranks):
        vec = []
        for j in range(nlocalranks):
            vec.append(localranks[(i + j) % nlocalranks])
        localrings[i] = vec
        
    return localrings
        
  def genringchannels(self, rank: int, type: GroupType) -> RingChannels:    
    ringchannels = defaultdict(lambda: defaultdict(list))
    localrings = self.gen_local_ring(rank, type)
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "No corresponding group information is generated, and there is an error in creating the ring channel.")
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    nNodes = gp_info.nNodes
    nlocalRanks = gp_info.nRanks // nNodes
    
    delta = gp_info.Ranks[nlocalRanks] - gp_info.Ranks[0] if nNodes > 1 else 0
    
    for ring_id, ring_ranks in localrings.items():
        prev = -1
        
        for i in range(nNodes):
            node_recv = ring_ranks[0] + i * delta
            node_send = ring_ranks[nlocalRanks-1] + i * delta
            
            for j in range(nlocalRanks):
                current = ring_ranks[j] + i * delta
                
                if j == nlocalRanks-1:
                    next_node = ring_ranks[0] + (i + 1) * delta
                else:
                    next_node = ring_ranks[j+1] + i * delta
                    
                ringchannels[ring_id][current] = [prev, next_node, node_recv, node_send]
                prev = current
                
        end_rank = ring_ranks[nlocalRanks-1] + (nNodes - 1) * delta
        ringchannels[ring_id][ring_ranks[0]][0] = end_rank
        ringchannels[ring_id][end_rank][1] = ring_ranks[0]
        
    self.Allringchannels[gp_idx] = ringchannels
    return ringchannels
        
  def gettreechannels(self, rank: int, type: GroupType) -> TreeChannels:    
    treechannels = defaultdict(dict)
    localrings = defaultdict(list)
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info and group ring channel, resulting in an error in gettreechannels.")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    
    if gp_idx in self.Alltreechannels:
        return self.Alltreechannels[gp_idx]
    
    nNodes = gp_info.nNodes
    nlocalRanks = gp_info.nRanks // nNodes
    localrings = self.gen_local_ring(rank, type)
    delta = gp_info.Ranks[nlocalRanks] - gp_info.Ranks[0] if nNodes > 1 else 0
    
    rings = defaultdict(list)
    for ring_id, local_ranks in localrings.items():
        for i in range(nNodes):
            for j in range(nlocalRanks):
                current = local_ranks[j] + i * delta
                rings[ring_id].append(current)
    
    roots = self.genInterDouBinTree(gp_info)
    
    allnode2ranks = defaultdict(lambda: defaultdict(list))
    for ring_id, ring_ranks in rings.items():
        nrankspernode = gp_info.nRanks // nNodes
        for i in range(gp_info.nNodes):
            for j in range(nrankspernode):
                allnode2ranks[ring_id][i].append(ring_ranks[i * nrankspernode + j])
    
    channel_id = 0
    
    for ring_id, node2ranks in allnode2ranks.items():
        for root in roots:
            treechannel = {}
            for rank in gp_info.Ranks:
                cur = ncclTree(-1, rank, -1, [])
                treechannel[rank] = cur
            
            self.ConnInterIntraTree(root, node2ranks, treechannel)
            treechannels[channel_id] = treechannel
            channel_id += 1
        
        self.Alltreechannels[gp_idx] = treechannels
    
    return treechannels
        
  def get_nvls_channels(self, rank: int, type: GroupType) -> TreeChannels:
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info and group ring channel, resulting in an error in get_nvls_channels.")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    nvlschannel = defaultdict(dict)
    
    if gp_info.nNodes > 1:
        NcclLog.write_log(NcclLogLevel.DEBUG, "%d", "error NVLS ALGO dont")
        return {}
    else:
        ranks = gp_info.Ranks
        NVswitch = gp_info.NVSwitchs[0]
        
        for i in range(len(ranks)):
            nvlschannel[0][ranks[i]] = ncclTree(-1, ranks[i], NVswitch, [])
        
        nvlschannel[0][len(ranks)] = ncclTree(-1, NVswitch, -1, ranks)
    
    self.AllNVLSchannels[gp_idx] = nvlschannel
    return nvlschannel
        

  def get_nvls_tree_channels(self, rank: int, type: GroupType) -> NVLStreechannels:    
    nvlstreechannels = defaultdict(lambda: defaultdict(list))
    localrings = defaultdict(list)
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info , resulting in an error in get_nvls_tree_channels.")
        return {}
    
    gp_idx = self.GroupIndex[(rank, type)]
    gp_info = self.AllGroups[gp_idx]
    
    if gp_idx in self.AllNVLStreechannels:
        return self.AllNVLStreechannels[gp_idx]
    
    roots = self.genInterDouBinTree(gp_info)
    nNodes = gp_info.nNodes
    nlocalRanks = gp_info.nRanks // nNodes
    localrings = self.gen_local_ring(rank, type)
    delta = gp_info.Ranks[nlocalRanks] - gp_info.Ranks[0] if nNodes > 1 else 0
    
    rings = defaultdict(list)
    for ring_id, local_ranks in localrings.items():
        for i in range(nNodes):
            for j in range(nlocalRanks):
                current = local_ranks[j] + i * delta
                rings[ring_id].append(current)
    
    allnode2ranks = defaultdict(lambda: defaultdict(list))
    for ring_id, ring_ranks in rings.items():
        nrankspernode = gp_info.nRanks // nNodes
        for i in range(gp_info.nNodes):
            for j in range(nrankspernode):
                allnode2ranks[ring_id][i].append(ring_ranks[i * nrankspernode + j])
    
    channel_id = 0
    node2ranks = allnode2ranks[0]
    
    for root in roots:
        for index in range(nlocalRanks):
            nvlstreechannel = defaultdict(list)
            nodencclchannlenodes = {}
            
            for i in range(nNodes):
                noderanks = node2ranks[i]
                intra_topo = [noderanks[index], gp_info.NVSwitchs[i]]
                intra_topo.extend(noderanks)
                
                NcclLog.write_log(NcclLogLevel.DEBUG, f" node  {i} intra_topo")
                for num in intra_topo:
                    NcclLog.write_log(NcclLogLevel.DEBUG, f" {num}")
                
                root_node = self.gen_nvls_tree_intra_channels(intra_topo, nvlstreechannel)
                nodencclchannlenodes[i] = root_node
            
            if rank == 0:
                for node_rank, nodes in nvlstreechannel.items():
                    NcclLog.write_log(NcclLogLevel.DEBUG, f" rank  {node_rank} nvls tree nodes ")
                    for i, node in enumerate(nodes):
                        NcclLog.write_log(NcclLogLevel.DEBUG, f" node  {i} rank  {node.rank}")
                        if node.up is not None:
                            NcclLog.write_log(NcclLogLevel.DEBUG, f" up  {node.up.rank}")
                        NcclLog.write_log(NcclLogLevel.DEBUG, " down ")
                        for down in node.down:
                            NcclLog.write_log(NcclLogLevel.DEBUG, f" {down.rank} ")
            
            self.gen_nvls_tree_inter_channels(root, nodencclchannlenodes, nvlstreechannel)
            nvlstreechannels[channel_id] = nvlstreechannel
            channel_id += 1
    
    self.AllNVLStreechannels[gp_idx] = nvlstreechannels
    return nvlstreechannels
        
  def gen_nvls_tree_intra_channels(self, intra_topo: List[int], 
                                    nvlstreechannel: Dict[int, List[ncclChannelNode]]) -> ncclChannelNode:
    root = ncclChannelNode(-1, intra_topo[0], None, [])
    nvlstreechannel[root.rank].append(root)
    
    nvswitch = ncclChannelNode(-1, intra_topo[1], root, [])
    nvlstreechannel[nvswitch.rank].append(nvswitch)
    root.down.append(nvswitch)
    
    for i in range(2, len(intra_topo)):
        leaf = ncclChannelNode(-1, intra_topo[i], nvswitch, [])
        nvswitch.down.append(leaf)
        nvlstreechannel[leaf.rank].append(leaf)
    
    return root
        
  def gen_nvls_tree_inter_channels(self, root: DoubleBinaryTreeNode, 
                                    nodencclchannlenodes: Dict[int, ncclChannelNode], 
                                    nvlstreechannel: Dict[int, List[ncclChannelNode]]) -> ncclChannelNode:
    NcclLog = MockNcclLog.get_instance()
    
    if root is None:
        return None
    else:
        NcclLog.write_log(NcclLogLevel.DEBUG, f"before root.right:  {root.right}")
        NcclLog.write_log(NcclLogLevel.DEBUG, f"before root.left:  {root.left}")
        
        if root.left is not None:
            NcclLog.write_log(NcclLogLevel.DEBUG, f"after root.left:  {root.left}")
            cur = nodencclchannlenodes[root.node]
            left = nodencclchannlenodes[root.left.node]
            cur.down.append(left)
            left.up = cur
            self.gen_nvls_tree_inter_channels(root.left, nodencclchannlenodes, nvlstreechannel)
        
        if root.right is not None:
            NcclLog.write_log(NcclLogLevel.DEBUG, f"after root.right:  {root.right}")
            cur = nodencclchannlenodes[root.node]
            right = nodencclchannlenodes[root.right.node]
            cur.down.append(right)
            right.up = cur
            self.gen_nvls_tree_inter_channels(root.right, nodencclchannlenodes, nvlstreechannel)
        

  def get_algo_proto_info(self, type: GroupType, rank: int, op: ComType, 
                           data_size: int) -> ncclInfo:
    NcclLog = MockNcclLog.get_instance()
    
    if (rank, type) not in self.GroupIndex:
        NcclLog.write_log(NcclLogLevel.ERROR, "There is no corresponding group info, resulting in an error with get_algo_proto_info.")
        return None
    
    gp_info = self.AllGroups[self.GroupIndex[(rank, type)]]

    ncclInfoName = "" 

    if type == GroupType.TP:
        ncclInfoName = "TP"
    elif type == GroupType.DP:
        ncclInfoName = "DP"
    elif type == GroupType.EP:
        ncclInfoName = "EP"
    elif type == GroupType.DP_EP:
        ncclInfoName = "DP_EP"

    
    ncclInfoName += f"_{int(op)}_{data_size}"
    
    if ncclInfoName in self.nccl_infos:
        return self.nccl_infos[ncclInfoName]
    else:
        NVLSenable = os.getenv("AS_NVLS_ENABLE") == "1"
        
        info = ncclInfo()
        info.nBytes = data_size
        info.nChannels = 0
        info.coll = op
        
        if op == ComType.All_Reduce:
            if type == GroupType.TP:
                if self.gpu_type in (GPUType.A100, GPUType.A800):
                    info.algorithm = NCCL_ALGO_RING
                elif self.gpu_type in (GPUType.H100, GPUType.H800):
                    if gp_info.nRanks >= 8 and NVLSenable:
                        info.algorithm = NCCL_ALGO_NVLS
                    else:
                        info.algorithm = NCCL_ALGO_RING
                else:
                    info.algorithm = NCCL_ALGO_RING
            else:
                info.algorithm = NCCL_ALGO_RING
        elif op in (ComType.All_Gather, ComType.Reduce_Scatter, ComType.All_to_All):
            info.algorithm = NCCL_ALGO_RING
        else:
            info.algorithm = NCCL_ALGO_RING
        
        info.protocol = NCCL_PROTO_UNDEF
        self.nccl_infos[ncclInfoName] = info
        return info