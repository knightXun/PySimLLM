import sys
import ns3
from collections import deque
import argparse
import os
from typing import Any, Dict, Tuple, Optional

# 假设以下为AstraSim相关类的Python绑定（需根据实际情况补充）
class AstraNetworkAPI:
    def __init__(self, rank: int):
        self.rank = rank

class AstraMemoryAPI:
    pass

class timespec_t:
    def __init__(self):
        self.time_val = 0.0

class sim_comm:
    pass

class sim_request:
    def __init__(self):
        self.flowTag = None  # 假设flowTag是自定义类型

class RecvPacketEventHadndlerData:
    def __init__(self):
        self.event = None  # 假设event是EventType枚举
        self.flowTag = None  # 假设flowTag是ncclFlowTag类型

class ncclFlowTag:
    def __init__(self):
        self.tag_id = 0
        self.channel_id = 0
        self.child_flow_id = -1
        self.current_flow_id = -1

class MockNcclLog:
    _instance = None
    LOG_LEVEL_INFO = "INFO"
    LOG_LEVEL_DEBUG = "DEBUG"

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_log_name(self, name: str):
        pass

    def writeLog(self, level: str, msg: str, *args):
        print(msg % args)

# 全局变量（需根据实际情况初始化）
receiver_pending_queue: Dict[Tuple[Tuple[int, int], int], ncclFlowTag] = {}
node_num: int = 0
switch_num: int = 0
link_num: int = 0
trace_num: int = 0
nvswitch_num: int = 0
gpus_per_server: int = 0
GPUType = int  # 假设为枚举类型
NVswitchs: list[int] = []


class ASTRASimNetwork(AstraNetworkAPI):
    def __init__(self, rank: int, npu_offset: int = 0):
        super().__init__(rank)
        self.npu_offset = npu_offset
        self.sim_event_queue = deque()
        self.nodeHash: Dict[Tuple[int, int], int] = {}  # 假设存储(node_id, direction) -> count
        self.sentHash: Dict[Tuple[int, Tuple[int, int]], Any] = {}  # 假设存储(tag, (src, dest)) -> task
        self.recvHash: Dict[Tuple[int, Tuple[int, int]], int] = {}  # 假设存储(tag, (src, dest)) -> count
        self.expeRecvHash: Dict[Tuple[int, Tuple[int, int]], Any] = {}  # 假设存储(tag, (src, dest)) -> task

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
        # 定义任务结构（用字典模拟C++的task1）
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
        # 假设SendFlow是ns3的函数（需根据实际实现）
        SendFlow(self.rank, dst, count, msg_handler, fun_arg, tag, request)
        return 0

    def sim_recv(self, buffer: Any, count: int, type: int, src: int, tag: int,
                 request: sim_request, msg_handler: callable, fun_arg: Any) -> int:
        NcclLog = MockNcclLog.getInstance()
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

        NcclLog.writeLog(MockNcclLog.LOG_LEVEL_DEBUG,
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
                NcclLog.writeLog(MockNcclLog.LOG_LEVEL_DEBUG,
                                "网络包后到，先进行注册 recvHash do not find expeRecvHash.new make src %d dest %d t.count: %d channel_id %d current_flow_id %d",
                                task["src"], task["dest"], task["count"], tag, flowTag.current_flow_id)
            else:
                existing_task = self.expeRecvHash[key]
                NcclLog.writeLog(MockNcclLog.LOG_LEVEL_DEBUG,
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
    NcclLog = MockNcclLog.getInstance()
    NcclLog.writeLog(MockNcclLog.LOG_LEVEL_INFO, "init SimAI.log")

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
        # 假设AstraSim::Sys构造函数参数与C++版本一致（需根据实际调整）
        systems[j] = AstraSim.Sys(
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

    # 启动工作负载
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
