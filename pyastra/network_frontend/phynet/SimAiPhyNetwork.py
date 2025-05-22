from system.AstraNetworkAPI import AstraNetworkAPI, timespec_t
from PhySimAi import PhyNetSim
from system.MockNcclLog import MockNcclLog, NcclLogLevel
import SimAiEntry
from system.BootStrapnet import local_rank

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
        return timespec_t(PhyNetSim.Now())

    def sim_schedule(self, delta, fun_ptr, fun_arg):
        nccl_log = MockNcclLog.get_instance()
        nccl_log.write_log(NcclLogLevel.DEBUG, 
                          f"SimAiPhyNetWork::sim_schedule local_rank {local_rank}")
        PhyNetSim.Schedule(delta.time_val, fun_ptr, fun_arg)

    def sim_send(self, buffer, count, dtype, dst, tag, request, msg_handler, fun_arg):
        nccl_log = MockNcclLog.get_instance()
        dst += self.npu_offset
        SimAiEntry.send_flow(self.rank, dst, count, msg_handler, fun_arg, tag, request)
        return 0

    def sim_recv(self, buffer, count, dtype, src, tag, request, msg_handler, fun_arg):
        return 0