import os
import sys

from system.AstraParamParse import UserParam, ModeType
from system.AstraNetworkAPI import AstraNetworkAPI
from system.AstraNetworkAPI import sim_comm, sim_request, timespec_t
from system.AstraMemoryAPI import AstraMemoryAPI
from system.AstraParamParse import  UserParam

from ns3.AstraSimNetwork import receiver_pending_queue
from ns3.common import node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server
from ns3.common import gpu_type
from ns3.common import NVswitchs

from ns3.entry import expeRecvHash
from ns3.entry import recvHash, sentHash, nodeHash, local_rank

from AstraSim import AnaSim
from system.Sys import Sys

all_gpus = []
ngpus_per_node = 0

nodeHash = {}

workloads = []
physical_dims = []


RESULT_PATH = "./results/"
WORKLOAD_PATH = ""


class AnalyticalNetWork:
    def __init__(self, _):
        pass


def main():
    param = UserParam.getInstance()
    if param.parseArg(len(sys.argv), sys.argv):
        print("-h,       --help                Help message")
        return -1
    
    param.mode = ModeType.ANALYTICAL

    physical_dims = [param.gpus]
    # AnaInit(argc, argv);
    using_num_gpus = 0
    all_gpu_num = param.gpus[0]
    for a in physical_dims:
        job_npus = 1
        for dim in a:
            job_npus *= dim
        using_num_gpus += job_npus

    node2nvswitch = {}
    for i in range(all_gpu_num):
        node2nvswitch[i] = all_gpu_num + i // param.net_work_param.gpus_per_server
        
    for i in range(all_gpu_num, all_gpu_num + param.net_work_param.nvswitch_num):
        node2nvswitch[i] = i
        param.net_work_param.NVswitchs.append(i)

    physical_dims[0][0] += param.net_work_param.nvswitch_num
    using_num_gpus += param.net_work_param.nvswitch_num

    queues_per_dim = [1] * len(physical_dims[0])
    job_npus = 1
    for dim in physical_dims[0]:
        job_npus *= dim

    analytical_network = AnalyticalNetWork(0)
    systems = Sys(
        analytical_network,
        None,
        0,
        0,
        1,
        physical_dims[0],
        queues_per_dim,
        "",
        WORKLOAD_PATH + param.workload,
        param.comm_scale,
        1,
        1,
        1,
        0,
        RESULT_PATH + param.res,
        "Analytical_test",
        True,
        False,
        param.net_work_param.gpu_type,
        param.gpus,
        param.net_work_param.NVswitchs,
        param.net_work_param.gpus_per_server
    )
    systems.nvswitch_id = node2nvswitch[0]
    systems.num_gpus = using_num_gpus - param.net_work_param.nvswitch_num

    systems.workload.fire()
    print("SimAI begin run Analytical")
    AnaSim.Run()
    AnaSim.Stop()
    AnaSim.Destroy()

    print("SimAI-Analytical finished.")
    return 0


if __name__ == "__main__":
    main()