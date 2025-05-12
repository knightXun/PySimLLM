# 定义常量
NCCL_NUM_ALGORITHMS = 6
NCCL_ALGO_UNDEF = -1
NCCL_ALGO_TREE = 0
NCCL_ALGO_RING = 1
NCCL_ALGO_COLLNET_DIRECT = 2
NCCL_ALGO_COLLNET_CHAIN = 3
NCCL_ALGO_NVLS = 4
NCCL_ALGO_NVLS_TREE = 5

NCCL_NUM_PROTOCOLS = 3
NCCL_PROTO_UNDEF = -1
NCCL_PROTO_LL = 0
NCCL_PROTO_LL128 = 1
NCCL_PROTO_SIMPLE = 2
NCCL_WORK_SIZE = 512

VOLTA_COMPCAP_IDX = 0
AMPERE_COMPCAP_IDX = 1
HOPPER_COMPCAP_IDX = 2
NCCL_TOPO_CPU_VENDOR_AMD = 2

NCCL_TOPO_CPU_ARCH_X86 = 1
NCCL_TOPO_CPU_ARCH_POWER = 2
NCCL_TOPO_CPU_VENDOR_INTEL = 1
NCCL_TOPO_CPU_VENDOR_AMD = 2

NCCL_TOPO_PATTERN_BALANCED_TREE = 1
NCCL_TOPO_PATTERN_SPLIT_TREE = 2
NCCL_TOPO_PATTERN_TREE = 3
NCCL_TOPO_PATTERN_RING = 4
NCCL_TOPO_PATTERN_NVLS = 5

NCCL_NUM_FUNCTIONS = 5
ncclFunc_t = {
    "ncclFuncBroadcast": 0,
    "ncclFuncReduce": 1,
    "ncclFuncAllGather": 2,
    "ncclFuncReduceScatter": 3,
    "ncclFuncAllReduce": 4,
    "ncclFuncSendRecv": 5,
    "ncclFuncSend": 6,
    "ncclFuncRecv": 7,
    "ncclNumFuncs": 8
}

# LL128最大带宽
llMaxBws = [
    [39.0, 39.0, 20.4],
    [87.7, 22.5, 19.0],
    [87.7, 22.5, 19.0]
]

# 基础延迟
baseLat = [
    [6.8, 14.0, 0],
    [6.6, 14.0, 8.4],
    [6.8, 14.0, 0],
    [6.8, 14.0, 0],
    [0, 0, 23.0],
    [0, 0, 23.0]
]

# 每个通道的最大环形LL128带宽
perChMaxRingLL128Bws = [
    [20.0, 20.0, 20.0],
    [20.0, 20.0, 20.0],
    [36.7, 36.7, 36.7]
]

# 每个通道的最大树形LL128带宽
perChMaxTreeLL128Bws = [
    [20.0, 20.0, 20.0],
    [20.0, 20.0, 20.0],
    [36.7, 36.7, 29.0]
]

# 每个通道的最大树形带宽
perChMaxTreeBws = [
    [26.5, 18.5, 10.0],
    [24.0, 23.6, 17.8],
    [38.7, 41.4, 36.0]
]

# 硬件延迟
NCCL_HW_NVLINK = 0
NCCL_HW_PCI = 1
NCCL_HW_NET = 2
hwLat = [
    [
        [0.6, 1.25, 4],
        [0.6, 1.9, 3.4],
        [0, 0, 8.0],
        [0, 0, 4.75],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [1.0, 1.9, 6],
        [1.0, 2.5, 5.7],
        [0, 0, 8.0],
        [0, 0, 8.0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [5.0, 8.5, 14],
        [2.7, 4.0, 14.0],
        [0, 0, 10.7],
        [0, 0, 14],
        [0, 0, 18],
        [0, 0, 19]
    ]
]

# 链路和路径类型
LINK_LOC = 0
LINK_NVL = 1
LINK_PCI = 3
LINK_SYS = 7
LINK_NET = 8
PCI_BW = 12.0

PATH_LOC = 0
PATH_NVL = 1
PATH_NVB = 2
PATH_PIX = 3
PATH_PXB = 4
PATH_PXN = 5
PATH_PHB = 6
PATH_SYS = 7
PATH_NET = 8
PATH_DIS = 9

# 向上取整函数
def DIVUP(x, y):
    return (x + y - 1) // y

def alignUp(x, a):
    return (x + a - 1) & (-a)

# 工作类型枚举
class ncclWorkType:
    ncclWorkTypeUnused = 0
    ncclWorkTypeColl = 1
    ncclWorkTypeP2p = 2
    ncclWorkTypeRegColl = 3

# 工作头结构体
class ncclWorkHeader:
    def __init__(self):
        self.workNext = 0
        self.doneAcks = 0
        self.funcIndex = 0
        self.isLast = 0
        self.inFifo = 0
        self.type = 0

# 工作元素结构体
class ncclWorkElem:
    def __init__(self):
        self.flagBits = 0
        self.isUsed = 0
        self.redOpArgIsPtr = 0
        self.regUsed = 0
        self.nWarps = 0
        self.direct = 0
        self.sendbuff = None
        self.recvbuff = None
        self.count = 0
        self.lastChunkSize = 0
        self.root = 0
        self.bid = 0
        self.nChannels = 0
        self.redOpArg = 0

NCCL_MAX_WORK_ELEMENTS = (NCCL_WORK_SIZE - alignUp(len(ncclWorkHeader().__dict__), alignof(ncclWorkElem())) // len(ncclWorkElem().__dict__))

# 树形校正因子
treeCorrectionFactor = [
    [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0],
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.7, 0.7, 0.8, 0.9, 0.9]
]

MAXCHANNELS = 10
NCCL_TOPO_MAX_NODES = 10
NCCL_MAX_TREE_ARITY = 2

# 拓扑图结构体
class ncclTopoGraph:
    def __init__(self):
        self.id = 0
        self.pattern = 0
        self.crossNic = 0
        self.collNet = 0
        self.minChannels = 0
        self.maxChannels = 0
        self.nChannels = 0
        self.bwIntra = 0
        self.bwInter = 0
        self.latencyInter = 0
        self.typeIntra = 0
        self.typeInter = 0
        self.sameChannels = 0
        self.nHops = 0
        self.intra = [0] * (MAXCHANNELS * NCCL_TOPO_MAX_NODES)
        self.inter = [0] * (MAXCHANNELS * 2)

    