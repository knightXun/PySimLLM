from astra_net import AstraNetworkAPI  # 假设基础类存在
from phy_sim_ai import PhyNetSim       # 假设模拟时间模块存在
from mock_nccl_log import MockNcclLog, NcclLogLevel  # 假设日志模块存在

class SimAiPhyNetwork(AstraNetworkAPI):
    def __init__(self, local_rank):
        super().__init__(local_rank)
        self.npu_offset = 0

    def sim_comm_size(self, comm, size):
        return 0

    def sim_finish(self):
        return 0

    def sim_time_resolution(self):
        return 0.0

    def sim_init(self, MEM):
        return 0

    def sim_get_time(self):
        class timespec_t:
            def __init__(self, time_val):
                self.time_val = time_val
        return timespec_t(PhyNetSim.Now())

    def sim_schedule(self, delta, fun_ptr, fun_arg):
        nccl_log = MockNcclLog.get_instance()
        nccl_log.write_log(NcclLogLevel.DEBUG, 
                          f"SimAiPhyNetWork::sim_schedule local_rank {local_rank}")
        PhyNetSim.Schedule(delta.time_val, fun_ptr, fun_arg)

    def sim_send(self, buffer, count, dtype, dst, tag, request, msg_handler, fun_arg):
        nccl_log = MockNcclLog.get_instance()
        dst += self.npu_offset
        self.send_flow(self.rank, dst, count, msg_handler, fun_arg, tag, request)
        return 0

    def sim_recv(self, buffer, count, dtype, src, tag, request, msg_handler, fun_arg):
        return 0

# 假设存在的全局变量
local_rank = 0

# 辅助函数（需要根据实际实现补充）
def send_flow(rank, dst, count, msg_handler, fun_arg, tag, request):
    """模拟网络流发送的逻辑实现"""
    pass