import abc
from enum import Enum


class ComputeMetaData:
    def __init__(self):
        self.compute_delay = 0


class ComputeAPI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute(self, M, K, N, msg_handler, fun_arg):
        pass

    @abc.abstractmethod
    def __del__(self):
        pass


class AstraMemoryAPI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def mem_read(self, size):
        pass

    @abc.abstractmethod
    def mem_write(self, size):
        pass

    @abc.abstractmethod
    def npu_mem_read(self, size):
        pass

    @abc.abstractmethod
    def npu_mem_write(self, size):
        pass

    @abc.abstractmethod
    def nic_mem_read(self, size):
        pass

    @abc.abstractmethod
    def nic_mem_write(self, size):
        pass

    def __del__(self):
        pass


class time_type_e(Enum):
    SE = 1
    MS = 2
    US = 3
    NS = 4
    FS = 5


class req_type_e(Enum):
    UINT8 = 1
    BFLOAT16 = 2
    FP32 = 3


class ncclFlowTag:
    def __init__(self, channel_id=-1, chunk_id=-1, current_flow_id=-1, child_flow_id=-1,
                 sender_node=-1, receiver_node=-1, flow_size=-1, pQps=None, tag_id=-1, nvls_on=False):
        self.channel_id = channel_id
        self.chunk_id = chunk_id
        self.current_flow_id = current_flow_id
        self.child_flow_id = child_flow_id
        self.sender_node = sender_node
        self.receiver_node = receiver_node
        self.flow_size = flow_size
        self.pQps = pQps
        self.tag_id = tag_id
        self.nvls_on = nvls_on


class sim_request:
    def __init__(self, srcRank=0, dstRank=0, tag=0, reqType=req_type_e.UINT8,
                 reqCount=0, vnet=0, layerNum=0, flowTag=ncclFlowTag()):
        self.srcRank = srcRank
        self.dstRank = dstRank
        self.tag = tag
        self.reqType = reqType
        self.reqCount = reqCount
        self.vnet = vnet
        self.layerNum = layerNum
        self.flowTag = flowTag


class timespec_t:
    def __init__(self, time_res=time_type_e.SE, time_val=0):
        self.time_res = time_res
        self.time_val = time_val


class MetaData:
    def __init__(self):
        self.timestamp = timespec_t()


class AstraNetworkAPI(metaclass=abc.ABCMeta):
    class BackendType(Enum):
        NotSpecified = 1
        Garnet = 2
        NS3 = 3
        Analytical = 4

    def __init__(self, rank):
        self.rank = rank
        self.enabled = True

    def get_backend_type(self):
        return self.BackendType.NotSpecified

    @abc.abstractmethod
    def sim_comm_size(self, comm, size):
        pass

    def sim_comm_get_rank(self):
        return self.rank

    def sim_comm_set_rank(self, rank):
        self.rank = rank
        return self.rank

    @abc.abstractmethod
    def sim_finish(self):
        pass

    @abc.abstractmethod
    def sim_time_resolution(self):
        pass

    @abc.abstractmethod
    def sim_init(self, MEM):
        pass

    @abc.abstractmethod
    def sim_get_time(self):
        pass

    @abc.abstractmethod
    def sim_schedule(self, delta, fun_ptr, fun_arg):
        pass

    @abc.abstractmethod
    def sim_send(self, buffer, count, type, dst, tag, request, msg_handler, fun_arg):
        pass

    @abc.abstractmethod
    def sim_recv(self, buffer, count, type, src, tag, request, msg_handler, fun_arg):
        pass

    def pass_front_end_report(self, astraSimDataAPI):
        return

    def get_BW_at_dimension(self, dim):
        return -1

    def __del__(self):
        pass

    