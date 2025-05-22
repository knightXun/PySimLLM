import sys
from collections import deque
import argparse
import os
from typing import Any, Dict, Tuple, Optional

from system.AstraParamParse import UserParam, ModeType
from system.AstraNetworkAPI import AstraNetworkAPI
from system.AstraNetworkAPI import sim_comm, sim_request, timespec_t
from system.AstraMemoryAPI import AstraMemoryAPI
from system.AstraParamParse import  UserParam

from common import node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server
from common import gpu_type
from common import NVswitchs

import ns3
from ns3.AstraSimNetwork import receiver_pending_queue
from ns3.entry import expeRecvHash
from ns3.entry import recvHash, sentHash

# from AstraSim import AnaSim
from system.Sys import Sys
from system.RecvPacketEventHandlerData import RecvPacketEventHandlerData
from system.MockNcclLog import MockNcclLog
from system.AstraNetworkAPI  import ncclFlowTag
from system.Common import GPUType
from entry import send_flow

import ns

class sim_event:
    def __init__(self, buffer=None, count=0, type=0, dst=0, tag=0, fnType=""):
        self.buffer = buffer    
        self.count = count     
        self.type = type       
        self.dst = dst        
        self.tag = tag     
        self.fnType = fnType 

class ASTRASimNetwork(AstraNetworkAPI):
    def __init__(self, rank: int, npu_offset: int = 0):
        super().__init__(rank)
        self.npu_offset = npu_offset
        self.sim_event_queue = deque()
        self.nodeHash: Dict[Tuple[int, int], int] = {}  
        self.sentHash: Dict[Tuple[int, Tuple[int, int]], Any] = {} 
        self.recvHash: Dict[Tuple[int, Tuple[int, int]], int] = {} 
        self.expeRecvHash: Dict[Tuple[int, Tuple[int, int]], Any] = {} 

    def sim_comm_size(self, comm: sim_comm, size: list[int]) -> int:
        return 0

    def sim_finish(self) -> int:
        for key in list(self.nodeHash.keys()):
            node_id, direction = key
            count = self.nodeHash[key]
            if direction == 0:
                print(f"sim_finish on sent, Thread id: {id(self)}")
                print(f"All data sent from node {node_id} is {count}")
            else:
                print(f"sim_finish on received, Thread id: {id(self)}")
                print(f"All data received by node {node_id} is {count}")
        os._exit(0)
        return 0

    def sim_time_resolution(self) -> float:
        return 0.0

    def sim_init(self, MEM: AstraMemoryAPI) -> int:
        return 0

    def sim_get_time(self) -> timespec_t:
        timeSpec = timespec_t()
        timeSpec.time_val = ns3.Simulator.Now().GetNanoSeconds()
        return timeSpec

    def sim_schedule(self, delta: timespec_t, fun_ptr: callable, fun_arg: Any) -> None:
        task = {
            "type": 2,
            "fun_arg": fun_arg,
            "msg_handler": fun_ptr,
            "schTime": delta.time_val
        }
        ns3.Simulator.Schedule(ns3.NanoSeconds(task["schTime"]), task["msg_handler"], task["fun_arg"])

    def sim_send(self, buffer: Any, count: int, type: int, dst: int, tag: int,
                 request: sim_request, msg_handler: callable, fun_arg: Any) -> int:
        dst += self.npu_offset
        task = {
            "src": self.rank,
            "dest": dst,
            "count": count,
            "type": 0,
            "fun_arg": fun_arg,
            "msg_handler": msg_handler
        }
        # 模拟C++中的临界区（Python中用简单锁替代）
        key = (tag, (task["src"], task["dest"]))
        self.sentHash[key] = task

        send_flow(self.rank, dst, count, msg_handler, fun_arg, tag, request)
        return 0

    def sim_recv(self, buffer: Any, count: int, type: int, src: int, tag: int,
                 request: sim_request, msg_handler: callable, fun_arg: Any) -> int:
        NcclLog = MockNcclLog.get_instance()
        flowTag = request.flowTag
        src += self.npu_offset
        task = {
            "src": src,
            "dest": self.rank,
            "count": count,
            "type": 1,
            "fun_arg": fun_arg,
            "msg_handler": msg_handler
        }
        ehd = task["fun_arg"]  # 假设是RecvPacketEventHadndlerData类型
        event = ehd.event
        tag = ehd.flowTag.tag_id

        NcclLog.write_log(MockNcclLog.LOG_LEVEL_DEBUG,
                        "接收事件注册 src %d sim_recv on rank %d tag_id %d channdl id %d",
                        src, self.rank, tag, ehd.flowTag.channel_id)

        key = (tag, (task["src"], task["dest"]))
        if key in self.recvHash:
            existing_count = self.recvHash[key]
            if existing_count == task["count"]:
                del self.recvHash[key]
                if (self.rank, src, tag) in receiver_pending_queue:
                    ehd.flowTag = receiver_pending_queue.pop((self.rank, src, tag))
                task["msg_handler"](task["fun_arg"])
            elif existing_count > task["count"]:
                self.recvHash[key] = existing_count - task["count"]
                if (self.rank, src, tag) in receiver_pending_queue:
                    ehd.flowTag = receiver_pending_queue.pop((self.rank, src, tag))
                task["msg_handler"](task["fun_arg"])
            else:
                del self.recvHash[key]
                task["count"] -= existing_count
                self.expeRecvHash[key] = task
        else:
            if key not in self.expeRecvHash:
                self.expeRecvHash[key] = task
                NcclLog.write_log(MockNcclLog.LOG_LEVEL_DEBUG,
                                "网络包后到，先进行注册 recvHash do not find expeRecvHash.new make src %d dest %d t.count: %d channel_id %d current_flow_id %d",
                                task["src"], task["dest"], task["count"], tag, flowTag.current_flow_id)
            else:
                existing_task = self.expeRecvHash[key]
                NcclLog.write_log(MockNcclLog.LOG_LEVEL_DEBUG,
                                "网络包后到，重复注册 recvHash do not find expeRecvHash.add make src %d dest %d expecount: %d t.count: %d tag_id %d current_flow_id %d",
                                task["src"], task["dest"], existing_task["count"], task["count"], tag, flowTag.current_flow_id)
        return 0

    def handleEvent(self, dst: int, cnt: int) -> None:
        pass


def user_param_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASTRASim Network Simulation")
    parser.add_argument("-t", "--thread", type=int, default=1, help="Number of threads")
    parser.add_argument("-w", "--workload", type=str, default="", help="Workload name")
    parser.add_argument("-n", "--network_topo", type=str, default="", help="Network topology file")
    parser.add_argument("-c", "--network_conf", type=str, default="", help="Network configuration file")
    return parser.parse_args()


def main():
    # 初始化日志
    MockNcclLog().set_log_name("SimAI.log")
    NcclLog = MockNcclLog.get_instance()
    NcclLog.write_log(MockNcclLog.LOG_LEVEL_INFO, "init SimAI.log")

    # 解析参数
    args = user_param_parse()

    # 初始化ns3 MTP接口（假设存在Python绑定）
    if "NS3_MTP" in os.environ:
        # 假设MtpInterface有Python绑定
        from ns3 import MtpInterface
        MtpInterface.Enable(args.thread)

    # 假设main1是网络拓扑初始化函数（需根据实际实现）
    main1(args.network_topo, args.network_conf)

    # 计算节点数（需根据实际全局变量初始化）
    nodes_num = node_num - switch_num
    gpu_num = node_num - nvswitch_num - switch_num

    # 初始化NVswitch映射
    node2nvswitch: Dict[int, int] = {}
    for i in range(gpu_num):
        node2nvswitch[i] = gpu_num + (i // gpus_per_server)
    for i in range(gpu_num, gpu_num + nvswitch_num):
        node2nvswitch[i] = i
        NVswitchs.append(i)

    # 启用ns3日志组件
    ns3.LogComponentEnable("OnOffApplication", ns3.LOG_LEVEL_INFO)
    ns3.LogComponentEnable("PacketSink", ns3.LOG_LEVEL_INFO)
    ns3.LogComponentEnable("GENERIC_SIMULATION", ns3.LOG_LEVEL_INFO)

    # 创建网络和系统实例
    networks: list[Optional[ASTRASimNetwork]] = [None] * nodes_num
    systems: list[Optional[Any]] = [None] * nodes_num  # 假设AstraSim::Sys有Python绑定

    for j in range(nodes_num):
        networks[j] = ASTRASimNetwork(j, 0)
        systems[j] = Sys(
            networks[j],
            None,
            j,
            0,
            1,
            [nodes_num],
            [1],
            "",
            args.workload,
            1,
            1,
            1,
            1,
            0,
            RESULT_PATH,
            "test1",
            True,
            False,
            gpu_type,
            [gpu_num],
            NVswitchs,
            gpus_per_server
        )
        systems[j].nvswitch_id = node2nvswitch[j]
        systems[j].num_gpus = nodes_num - nvswitch_num

    for i in range(nodes_num):
        if systems[i] is not None and systems[i].workload is not None:
            systems[i].workload.fire()

    print("simulator run")
    ns3.Simulator.Run()
    ns3.Simulator.Stop(ns3.Seconds(2000000000))
    ns3.Simulator.Destroy()

    # 关闭MTP接口（如果启用）
    if "NS3_MPI" in os.environ:
        from ns3 import MpiInterface
        MpiInterface.Disable()


if __name__ == "__main__":
    main()
