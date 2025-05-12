import os
import sys
import argparse
from typing import List, Dict

# 模拟原C++中的全局变量
local_rank = 0  # 需根据实际运行环境设置，这里设为默认值
global_sys = None
flow_rdma = None  # 假设后续会根据条件初始化

RESULT_PATH = "/etc/astra-sim/results/ncclFlowModel_"

class GPUType:
    A100 = "A100"  # 用枚举类或简单常量模拟原枚举

class UserParam:
    """模拟原user_param结构体"""
    def __init__(self):
        self.thread = 1
        self.gpus = 8
        self.workload = "microAllReduce.txt"
        self.comm_scale = 1
        self.gpu_type = GPUType.A100
        self.nvswitch_num = 1
        self.gpus_per_server = 8
        self.gid_index = 0

    def __repr__(self):
        return (f"UserParam(thread={self.thread}, gpus={self.gpus}, "
                f"workload='{self.workload}', comm_scale={self.comm_scale}, "
                f"gpu_type={self.gpu_type}, nvswitch_num={self.nvswitch_num}, "
                f"gpus_per_server={self.gpus_per_server}, gid_index={self.gid_index})")

def parse_user_params(args: List[str]) -> UserParam:
    """解析命令行参数（替代原user_param_prase函数）"""
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

class SimAiPhyNetwork:
    """模拟原SimAiPhyNetWork类"""
    def __init__(self, local_rank: int):
        self.local_rank = local_rank

class AstraSys:
    """模拟原AstraSim::Sys类"""
    def __init__(self, phy_network, local_rank: int, physical_dims: List[int], 
                 queues_per_dim: List[int], workload: str, comm_scale: float, 
                 result_path: str, gpu_type: GPUType, num_gpus: int, 
                 nvswitchs: List[int], gpus_per_server: int):
        self.phy_network = phy_network
        self.local_rank = local_rank
        self.physical_dims = physical_dims
        self.queues_per_dim = queues_per_dim
        self.workload = workload
        self.comm_scale = comm_scale
        self.result_path = result_path
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus
        self.nvswitchs = nvswitchs
        self.gpus_per_server = gpus_per_server
        self.nvswitch_id = None  # 后续赋值

class MockNcclLog:
    """模拟原MockNcclLog单例类"""
    _instance = None
    log_levels = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "ERROR": "ERROR"
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.log_name = ""
        return cls._instance

    def set_log_name(self, name: str):
        self.log_name = name

    def write_log(self, level: str, message: str, *args):
        with open(self.log_name, "a") as f:
            f.write(f"[{level}] {message % args}\n")

def boot_strap_net(argc: int, argv: List[str]):
    """模拟原BootStrapNet函数"""
    # 实际应包含网络初始化逻辑，这里简化
    pass

def set_simai_network_callback():
    """模拟原网络回调设置函数"""
    # 实际应包含回调注册逻辑，这里简化
    pass

def phy_net_sim_run():
    """模拟原PhyNetSim::Run()"""
    # 实际应包含模拟运行逻辑，这里简化
    print("PhyNetSim running...")

def phy_net_sim_stop():
    """模拟原PhyNetSim::Stop()"""
    # 实际应包含模拟停止逻辑，这里简化
    print("PhyNetSim stopped.")

def notify_all_thread_finished():
    """模拟原线程通知函数"""
    # 实际应包含线程同步逻辑，这里简化
    print("All threads notified.")

def phy_net_sim_destroy():
    """模拟原PhyNetSim::Destory()"""
    # 实际应包含资源释放逻辑，这里简化
    print("PhyNetSim destroyed.")

def main():
    # 模拟原main函数流程
    boot_strap_net(len(sys.argv), sys.argv)
    pid = os.getpid()
    
    # 初始化日志
    nccl_log = MockNcclLog()
    nccl_log.set_log_name(f"SimAi_{local_rank}.log")
    nccl_log.write_log(MockNcclLog.log_levels["DEBUG"], "Local rank %d PID %d", local_rank, pid)
    
    # 解析参数
    user_params = parse_user_params(sys.argv[1:])
    print(f"解析到参数: {user_params}")
    
    # 模拟RDMA初始化（条件编译转换为条件判断）
    if "PHY_RDMA" in os.environ:  # 假设通过环境变量控制是否启用RDMA
        global flow_rdma
        # 这里需要根据实际需求实现FlowPhyRdma类
        class FlowPhyRdma:
            def ibv_init(self):
                print("RDMA初始化完成")
        flow_rdma = FlowPhyRdma()
        flow_rdma.ibv_init()
    
    # 设置网络回调
    set_simai_network_callback()
    
    # 构建物理维度和NVSwitch信息
    physical_dims = [user_params.gpus]
    nvswitchs: List[int] = []
    node2nvswitch: Dict[int, int] = {}
    
    for i in range(user_params.gpus):
        node2nvswitch[i] = user_params.gpus + (i // user_params.gpus_per_server)
    
    for i in range(user_params.gpus, user_params.gpus + user_params.nvswitch_num):
        node2nvswitch[i] = i
        nvswitchs.append(i)
    
    physical_dims[0] += user_params.nvswitch_num  # 更新物理维度
    
    # 初始化物理网络和全局系统
    phy_network = SimAiPhyNetwork(local_rank)
    global global_sys
    global_sys = AstraSys(
        phy_network=phy_network,
        local_rank=local_rank,
        physical_dims=physical_dims,
        queues_per_dim=[1],
        workload=user_params.workload,
        comm_scale=user_params.comm_scale,
        result_path=RESULT_PATH,
        gpu_type=user_params.gpu_type,
        num_gpus=user_params.gpus,
        nvswitchs=nvswitchs,
        gpus_per_server=user_params.gpus_per_server
    )
    global_sys.nvswitch_id = node2nvswitch[local_rank]
    global_sys.num_gpus = user_params.gpus
    
    # 运行模拟
    if global_sys.workload:
        print(f"启动工作负载: {global_sys.workload}")  # 实际应实现workload.fire()
    
    phy_net_sim_run()
    phy_net_sim_stop()
    notify_all_thread_finished()
    phy_net_sim_destroy()
    
    # 注意：原MPI_Finalize()需要安装mpi4py库，这里注释掉需要时启用
    # from mpi4py import MPI
    # MPI.Finalize()

if __name__ == "__main__":
    main()