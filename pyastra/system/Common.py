import enum

# 这里假设 AstraNetworkAPI 有对应的 Python 模块，如果没有则需要替换为实际可用的代码
# 由于没有具体内容，这里暂时用空类表示
class AstraNetworkAPI:
    pass

class GPUType(enum.Enum):
    A100 = 0
    A800 = 1
    H100 = 2
    H800 = 3
    NONE = 4

CLOCK_PERIOD = 1
FREQ = 1000.0 / CLOCK_PERIOD
GBps = 1.0 / (1024 * 1024 * 1024)
Tick = int

class ComType(enum.Enum):
    None = 0
    Reduce_Scatter = 1
    All_Gather = 2
    All_Reduce = 3
    All_to_All = 4
    All_Reduce_All_to_All = 5
    All_Reduce_NVLS = 6

class CollectiveOptimization(enum.Enum):
    Baseline = 0
    LocalBWAware = 1

class CollectiveImplementationType(enum.Enum):
    Ring = 0
    OneRing = 1
    Direct = 2
    OneDirect = 3
    AllToAll = 4
    DoubleBinaryTreeLocalAllToAll = 5
    LocalRingNodeA2AGlobalDBT = 6
    HierarchicalRing = 7
    DoubleBinaryTree = 8
    HalvingDoubling = 9
    OneHalvingDoubling = 10
    NcclFlowModel = 11
    NcclTreeFlowModel = 12

class CollectiveBarrier(enum.Enum):
    Blocking = 0
    Non_Blocking = 1

class SchedulingPolicy(enum.Enum):
    LIFO = 0
    FIFO = 1
    HIGHEST = 2
    None = 3

class IntraDimensionScheduling(enum.Enum):
    FIFO = 0
    RG = 1
    SmallestFirst = 2
    LessRemainingPhaseFirst = 3

class InterDimensionScheduling(enum.Enum):
    Ascending = 0
    OnlineGreedy = 1
    RoundRobin = 2
    OfflineGreedy = 3
    OfflineGreedyFlex = 4

class InjectionPolicy(enum.Enum):
    Infinite = 0
    Aggressive = 1
    SemiAggressive = 2
    ExtraAggressive = 3
    Normal = 4

class PacketRouting(enum.Enum):
    Hardware = 0
    Software = 1

class BusType(enum.Enum):
    Both = 0
    Shared = 1
    Mem = 2

class StreamState(enum.Enum):
    Created = 0
    Transferring = 1
    Ready = 2
    Executing = 3
    Zombie = 4
    Dead = 5

class EventType(enum.Enum):
    NONE = 0
    RendezvousSend = 1
    RendezvousRecv = 2
    CallEvents = 3
    PacketReceived = 4
    PacketSent = 5
    PacketSentFinshed = 6
    WaitForVnetTurn = 7
    NCCL_General = 8
    General = 9
    TX_DMA = 10
    RX_DMA = 11
    Wight_Grad_Comm_Finished = 12
    Input_Grad_Comm_Finished = 13
    Fwd_Comm_Finished = 14
    Wight_Grad_Comm_Finished_After_Delay = 15
    Input_Grad_Comm_Finished_After_Delay = 16
    Fwd_Comm_Finished_After_Delay = 17
    Workload_Wait = 18
    Reduction_Ready = 19
    Rec_Finished = 20
    Send_Finished = 21
    Processing_Finished = 22
    Delivered = 23
    NPU_to_MA = 24
    MA_to_NPU = 25
    Read_Port_Free = 26
    Write_Port_Free = 27
    Apply_Boost = 28
    Stream_Transfer_Started = 29
    Stream_Ready = 30
    Consider_Process = 31
    Consider_Retire = 32
    Consider_Send_Back = 33
    StreamInit = 34
    StreamsFinishedIncrease = 35
    CommProcessingFinished = 36
    NotInitialized = 37

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
    