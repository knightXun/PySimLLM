import sys
from collections import deque
import argparse
import os
from typing import Any, Dict, Tuple, Optional

# import ns

def user_param_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySimLLM Simulation")
    parser.add_argument("-w", "--workload", type=str, default="", help="Workload name")
    parser.add_argument("-n", "--network_topo", type=str, default="", help="Network topology file")
    parser.add_argument("-c", "--network_conf", type=str, default="", help="Network configuration file")
    return parser.parse_args()


def main():
    args = user_param_parse()
    from ns import ns 
    import net
    from net import ReadConf, SetConfig, SetupNetwork
    from workload import Workload

    ReadConf(args.network_topo, args.network_conf)
    print("Read Conf Done.")
    SetConfig()
    SetupNetwork(None, None) 

    from FlowModel import FlowModel
    gpu_servers = net.node_num - net.nvswitch_num - net.switch_num
    nodes = list(range(gpu_servers))
    NVswitchs = list(range(gpu_servers,gpu_servers + net.nvswitch_num))

    flowModel = FlowModel(nodes, NVswitchs, net.n, \
        net.portNumber, net.pairBdp, net.has_win, \
        net.global_t, net.pairRtt, net.maxRtt, \
        net.serverAddress, net.maxBdp)

    print("Running Simulation.")
    
    w = Workload("test", args.workload, 1, 1, 0, "etc", False, gpu_servers, flowModel)
    w.run()

    print(f"{gpu_servers} nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
    ns.Simulator.Destroy()
    print("Simulation Done.")

if __name__ == "__main__":
    main()
