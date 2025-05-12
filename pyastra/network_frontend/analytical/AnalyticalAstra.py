import os
import sys

# 假设的全局变量和映射
receiver_pending_queue = {}
node_num = 0
switch_num = 0
link_num = 0
trace_num = 0
nvswitch_num = 0
gpus_per_server = 0
gpu_type = ""
NVswitchs = []
all_gpus = []
ngpus_per_node = 0
expeRecvHash = {}
recvHash = {}
sentHash = {}
nodeHash = {}
local_rank = 0

workloads = []
physical_dims = []

class UserParam:
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.thread = 1
        self.gpus = [1]
        self.workload = ""
        self.comm_scale = 1
        self.mode = None
        self.net_work_param = type('NetWorkParam', (object,), {
            'gpus_per_server': 0,
            'nvswitch_num': 0,
            'NVswitchs': [],
            'gpu_type': ""
        })()
        self.res = ""

    def parseArg(self, argc, argv):
        if "-h" in argv or "--help" in argv:
            return True
        # 这里应该添加实际的参数解析逻辑
        return False


RESULT_PATH = "./results/"
WORKLOAD_PATH = ""


class AnalyticalNetWork:
    def __init__(self, _):
        pass


class Sys:
    def __init__(self, analytical_network, _, __, ___, ____, physical_dim, queues_per_dim, _____, workload_path,
                 comm_scale, ______, _______, ________, _________, result_path, __________, ___________, ____________,
                 gpu_type, gpus, NVswitchs, gpus_per_server):
        self.analytical_network = analytical_network
        self.workload = type('Workload', (object,), {'fire': lambda: None})()
        self.nvswitch_id = None
        self.num_gpus = None
        self.net_work_param = type('NetWorkParam', (object,), {
            'gpus_per_server': gpus_per_server,
            'NVswitchs': NVswitchs,
            'gpu_type': gpu_type
        })()
        self.gpus = gpus
        self.comm_scale = comm_scale
        self.result_path = result_path
        self.workload_path = workload_path


class AnaSim:
    @staticmethod
    def Run():
        pass

    @staticmethod
    def Stop():
        pass

    @staticmethod
    def Destroy():
        pass


def main():
    param = UserParam.getInstance()
    if param.parseArg(len(sys.argv), sys.argv):
        print("-h,       --help                Help message")
        return -1
    param.mode = "ANALYTICAL"
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