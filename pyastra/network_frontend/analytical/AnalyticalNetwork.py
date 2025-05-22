from system.AstraNetworkAPI import AstraNetworkAPI
from system.AstraNetworkAPI import sim_comm, sim_request, timespec_t
from system.AstraMemoryAPI import AstraMemoryAPI

from ns3.AstraSimNetwork import receiver_pending_queue
from ns3.common import node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server
from ns3.common import gpu_type
from ns3.common import NVswitchs

from ns3.entry import expeRecvHash
from ns3.entry import recvHash, sentHash, nodeHash, local_rank

from AstraSim import AnaSim

class AnalyticalNetWork(AstraNetworkAPI):
    def __init__(self, _local_rank: int):
        super().__init__(_local_rank)
        self.npu_offset = 0  

    def __del__(self):
        pass

    def sim_comm_size(self, comm: sim_comm, size: list) -> int:
        # size参数用列表模拟指针传递（Python中列表是可变对象）
        return 0

    def sim_finish(self) -> int:
        return 0

    def sim_time_resolution(self) -> float:
        return 0.0

    def sim_init(self, MEM: AstraMemoryAPI) -> int:
        return 0

    def sim_get_time(self) -> timespec_t:
        time_spec = timespec_t()
        time_spec.time_val = AnaSim.Now()
        return time_spec

    def sim_schedule(self, delta: timespec_t, fun_ptr, fun_arg) -> None:
        AnaSim.Schedule(delta.time_val, fun_ptr, fun_arg)

    def sim_send(
        self,
        buffer,
        count: int,
        type_: int, 
        dst: int,
        tag: int,
        request: sim_request,
        msg_handler,
        fun_arg
    ) -> int:
        return 0

    def sim_recv(
        self,
        buffer,
        count: int,
        type_: int,  # 避免与Python内置type冲突
        src: int,
        tag: int,
        request: sim_request,
        msg_handler,
        fun_arg
    ) -> int:
        return 0