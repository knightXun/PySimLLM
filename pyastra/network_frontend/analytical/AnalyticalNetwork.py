# 假设AstraSim模块已正确导入并包含相关基础类定义
import AstraSim

# 外部全局变量（对应C++中的extern声明）
receiver_pending_queue = {}  # 键类型：tuple[tuple[tuple[int, int], int], ...]
expeRecvHash = {}            # 键类型：tuple[int, tuple[int, int]]
recvHash = {}                # 键类型：tuple[int, tuple[int, int]]
sentHash = {}                # 键类型：tuple[int, tuple[int, int]]
nodeHash = {}                # 键类型：tuple[int, int]
local_rank = 0               # 全局rank变量

class AnalyticalNetWork(AstraSim.AstraNetworkAPI):
    def __init__(self, _local_rank: int):
        super().__init__(_local_rank)
        self.npu_offset = 0  # 私有成员变量

    def __del__(self):
        # 对应C++析构函数（Python自动垃圾回收，通常无需显式定义）
        pass

    def sim_comm_size(self, comm: AstraSim.sim_comm, size: list) -> int:
        # size参数用列表模拟指针传递（Python中列表是可变对象）
        return 0

    def sim_finish(self) -> int:
        return 0

    def sim_time_resolution(self) -> float:
        return 0.0

    def sim_init(self, MEM: AstraSim.AstraMemoryAPI) -> int:
        return 0

    def sim_get_time(self) -> AstraSim.timespec_t:
        time_spec = AstraSim.timespec_t()
        time_spec.time_val = AstraSim.AnaSim.Now()
        return time_spec

    def sim_schedule(self, delta: AstraSim.timespec_t, fun_ptr, fun_arg) -> None:
        AstraSim.AnaSim.Schedule(delta.time_val, fun_ptr, fun_arg)

    def sim_send(
        self,
        buffer,
        count: int,
        type_: int,  # 避免与Python内置type冲突
        dst: int,
        tag: int,
        request: AstraSim.sim_request,
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
        request: AstraSim.sim_request,
        msg_handler,
        fun_arg
    ) -> int:
        return 0