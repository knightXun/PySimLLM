import enum

# 这里假设 AstraNetworkAPI 有对应的 Python 模块，如果没有则需要替换为实际可用的代码
# 由于没有具体内容，这里暂时用空类表示
class AstraNetworkAPI:
    pass

class GPUType(enum.Enum):
    A100 = 'A100'
    A800 = 'A800'
    H100 = 'H100'
    H800 = 'H800'
    NONE = 'NONE'

CLOCK_PERIOD = 1
FREQ = 1000.0 / CLOCK_PERIOD
GBps = 1.0 / (1024 * 1024 * 1024)
Tick = int

class ComType(enum.Enum):
    None = 'None'
    Reduce_Scatter = 'Reduce_Scatter'
    All_Gather = 'All_Gather'
    All_Reduce = 'All_Reduce'
    All_to_All = 'All_to_All'
    All_Reduce_All_to_All = 'All_Reduce_All_to_All'
    All_Reduce_NVLS = 'All_Reduce_NVLS'

class CollectiveOptimization(enum.Enum):
    Baseline = 'Baseline'
    LocalBWAware = 'LocalBWAware'

class CollectiveImplementationType(enum.Enum):
    Ring = 'Ring'
    OneRing = 'OneRing'
    Direct = 'Direct'
    OneDirect = 'OneDirect'
    AllToAll = 'AllToAll'
    DoubleBinaryTreeLocalAllToAll = 'DoubleBinaryTreeLocalAllToAll'
    LocalRingNodeA2AGlobalDBT = 'LocalRingNodeA2AGlobalDBT'
    HierarchicalRing = 'HierarchicalRing'
    DoubleBinaryTree = 'DoubleBinaryTree'
    HalvingDoubling = 'HalvingDoubling'
    OneHalvingDoubling = 'OneHalvingDoubling'
    NcclFlowModel = 'NcclFlowModel'
    NcclTreeFlowModel = 'NcclTreeFlowModel'

class CollectiveBarrier(enum.Enum):
    Blocking = 'Blocking'
    Non_Blocking = 'Non_Blocking'

class SchedulingPolicy(enum.Enum):
    LIFO = 'LIFO'
    FIFO = 'FIFO'
    HIGHEST = 'HIGHEST'
    None = 'None'

class IntraDimensionScheduling(enum.Enum):
    FIFO = 'FIFO'
    RG = 'RG'
    SmallestFirst = 'SmallestFirst'
    LessRemainingPhaseFirst = 'LessRemainingPhaseFirst'

class InterDimensionScheduling(enum.Enum):
    Ascending = 'Ascending'
    OnlineGreedy = 'OnlineGreedy'
    RoundRobin = 'RoundRobin'
    OfflineGreedy = 'OfflineGreedy'
    OfflineGreedyFlex = 'OfflineGreedyFlex'

class InjectionPolicy(enum.Enum):
    Infinite = 'Infinite'
    Aggressive = 'Aggressive'
    SemiAggressive = 'SemiAggressive'
    ExtraAggressive = 'ExtraAggressive'
    Normal = 'Normal'

class PacketRouting(enum.Enum):
    Hardware = 'Hardware'
    Software = 'Software'

class BusType(enum.Enum):
    Both = 'Both'
    Shared = 'Shared'
    Mem = 'Mem'

class StreamState(enum.Enum):
    Created = 'Created'
    Transferring = 'Transferring'
    Ready = 'Ready'
    Executing = 'Executing'
    Zombie = 'Zombie'
    Dead = 'Dead'

class EventType(enum.Enum):
    NONE = 'NONE'
    RendezvousSend = 'RendezvousSend'
    RendezvousRecv = 'RendezvousRecv'
    CallEvents = 'CallEvents'
    PacketReceived = 'PacketReceived'
    PacketSent = 'PacketSent'
    PacketSentFinshed = 'PacketSentFinshed'
    WaitForVnetTurn = 'WaitForVnetTurn'
    NCCL_General = 'NCCL_General'
    General = 'General'
    TX_DMA = 'TX_DMA'
    RX_DMA = 'RX_DMA'
    Wight_Grad_Comm_Finished = 'Wight_Grad_Comm_Finished'
    Input_Grad_Comm_Finished = 'Input_Grad_Comm_Finished'
    Fwd_Comm_Finished = 'Fwd_Comm_Finished'
    Wight_Grad_Comm_Finished_After_Delay = 'Wight_Grad_Comm_Finished_After_Delay'
    Input_Grad_Comm_Finished_After_Delay = 'Input_Grad_Comm_Finished_After_Delay'
    Fwd_Comm_Finished_After_Delay = 'Fwd_Comm_Finished_After_Delay'
    Workload_Wait = 'Workload_Wait'
    Reduction_Ready = 'Reduction_Ready'
    Rec_Finished = 'Rec_Finished'
    Send_Finished = 'Send_Finished'
    Processing_Finished = 'Processing_Finished'
    Delivered = 'Delivered'
    NPU_to_MA = 'NPU_to_MA'
    MA_to_NPU = 'MA_to_NPU'
    Read_Port_Free = 'Read_Port_Free'
    Write_Port_Free = 'Write_Port_Free'
    Apply_Boost = 'Apply_Boost'
    Stream_Transfer_Started = 'Stream_Transfer_Started'
    Stream_Ready = 'Stream_Ready'
    Consider_Process = 'Consider_Process'
    Consider_Retire = 'Consider_Retire'
    Consider_Send_Back = 'Consider_Send_Back'
    StreamInit = 'StreamInit'
    StreamsFinishedIncrease = 'StreamsFinishedIncrease'
    CommProcessingFinished = 'CommProcessingFinished'
    NotInitialized = 'NotInitialized'

class CloneInterface:
    def clone(self):
        raise NotImplementedError("Subclasses should implement this!")

    def __del__(self):
        pass

class CollectiveImplementation(CloneInterface):
    def __init__(self, type):
        self.type = type

    def clone(self):
        return CollectiveImplementation(self.type)

class DirectCollectiveImplementation(CollectiveImplementation):
    def __init__(self, type, direct_collective_window):
        super().__init__(type)
        self.direct_collective_window = direct_collective_window

    def clone(self):
        return DirectCollectiveImplementation(self.type, self.direct_collective_window)
    