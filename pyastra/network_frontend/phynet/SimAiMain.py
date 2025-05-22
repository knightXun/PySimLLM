import os
import sys
import argparse
from typing import List, Dict

# from system.AstraParamParse import UserParam
from system.Common import GPUType
from system.BootStrapnet import local_rank, world_size
from system.SimAiFlowModelRdma import flow_rdma
from SimAiEntry import global_sys, set_simai_network_callback
from SimAiPhyNetwork import SimAiPhyNetwork
from system.Sys import Sys 
from system.MockNcclLog import MockNcclLog
from system.BootStrapnet import BootStrapNet
from system.SimAiFlowModelRdma import FlowPhyRdma
from PhySimAi import PhyNetSim
from system.PhyMultiThread import notify_all_thread_finished
from mpi4py import MPI

RESULT_PATH = "/etc/astra-sim/results/ncclFlowModel_"

class UserParam:
    def __init__(self, thread=1, gpus=8, workload="microAllReduce.txt", comm_scale=1, 
                 gpu_type=GPUType.A100, nvswitch_num=1, gpus_per_server=8, gid_index=0):
        self.thread = thread
        self.gpus = gpus
        self.workload = workload
        self.comm_scale = comm_scale
        self.gpu_type = gpu_type
        self.nvswitch_num = nvswitch_num
        self.gpus_per_server = gpus_per_server
        self.gid_index = gid_index


def parse_user_params(args: List[str]) -> UserParam:
    parser = argparse.ArgumentParser(description="AstraSim网络模拟参数解析")
    parser.add_argument('-t', '--thread', type=int, default=1, help='线程数')
    parser.add_argument('-w', '--workload', type=str, default="microAllReduce.txt", 
                       help='工作负载文件，默认microAllReduce.txt')
    parser.add_argument('-g', '--gpus', type=int, default=8, 
                       help='GPU数量，默认8（最小值8）')
    parser.add_argument('-s', '--comm_scale', type=float, default=1.0, 
                       help='通信规模因子，默认1')
    parser.add_argument('-i', '--gid_index', type=int, default=0, 
                       help='RDMA GID索引，默认0')
    
    args = parser.parse_args(args)
    params = UserParam()
    
    # 参数校验和赋值
    params.thread = args.thread
    params.workload = args.workload
    params.gpus = max(args.gpus, 8)  # 确保不小于8
    params.comm_scale = args.comm_scale
    params.gid_index = args.gid_index
    
    return params



def main():
    BootStrapNet(len(sys.argv), sys.argv)
    pid = os.getpid()
    
    # 初始化日志
    nccl_log = MockNcclLog.get_instance()
    nccl_log.set_log_name(f"SimAi_{local_rank}.log")
    nccl_log.write_log(MockNcclLog.log_levels["DEBUG"], "Local rank %d PID %d", local_rank, pid)
    
    # 解析参数
    user_params = parse_user_params(sys.argv[1:])
    print(f"解析到参数: {user_params}")
    
    if "PHY_RDMA" in os.environ: 
        flow_rdma = FlowPhyRdma()
        flow_rdma.ibv_init()
    
    # 设置网络回调
    set_simai_network_callback()
    
    # 构建物理维度和NVSwitch信息
    physical_dims = [user_params.gpus]
    NVswitchs: List[int] = []
    node2nvswitch: Dict[int, int] = {}
    queues_per_dim: List[int] = []

    for i in range(user_params.gpus):
        node2nvswitch[i] = user_params.gpus + (i // user_params.gpus_per_server)
    
    for i in range(user_params.gpus, user_params.gpus + user_params.nvswitch_num):
        node2nvswitch[i] = i
        NVswitchs.append(i)
    
    physical_dims[0] += user_params.nvswitch_num  # 更新物理维度
    
    # 初始化物理网络和全局系统
    phy_network = SimAiPhyNetwork(local_rank)
    global global_sys
    global_sys = Sys(
        phy_network,
        None,
        local_rank,
        0,
        1,
        physical_dims,
        queues_per_dim,
        "",
        user_params.workload,
        user_params.comm_scale,
        1,
        1,
        1,
        0,
        RESULT_PATH,
        "phynet_test",
        True,
        False,
        user_params.gpu_type,
        {user_params.gpus},
        NVswitchs,
        user_params.gpus_per_server
    )
    
    global_sys.nvswitch_id = node2nvswitch[local_rank]
    global_sys.num_gpus = user_params.gpus
    global_sys.workload.fire()

    
    PhyNetSim.run()
    PhyNetSim.stop()
    notify_all_thread_finished()
    PhyNetSim.destroy()
    
    MPI.Finalize()

if __name__ == "__main__":
    main()