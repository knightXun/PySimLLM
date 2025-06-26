import sys
from collections import deque
import argparse
import os
from typing import Any, Dict, Tuple, Optional

import ns

from common import node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server
from common import gpu_type
from common import NVswitchs
import common

def user_param_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASTRASim Network Simulation")
    parser.add_argument("-t", "--thread", type=int, default=1, help="Number of threads")
    parser.add_argument("-w", "--workload", type=str, default="", help="Workload name")
    parser.add_argument("-n", "--network_topo", type=str, default="", help="Network topology file")
    parser.add_argument("-c", "--network_conf", type=str, default="", help="Network configuration file")
    return parser.parse_args()


def main():

    args = user_param_parse()

    # 假设main1是网络拓扑初始化函数（需根据实际实现）
    common.main1(args.network_topo, args.network_conf)

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
    ns.Simulator.Run()
    ns.Simulator.Stop(ns.Seconds(2000000000))
    ns.Simulator.Destroy()


if __name__ == "__main__":
    main()
