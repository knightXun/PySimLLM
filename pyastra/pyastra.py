import sys
from collections import deque
import argparse
import os
from typing import Any, Dict, Tuple, Optional

import ns

def user_param_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySimLLM Simulation")
    parser.add_argument("-w", "--workload", type=str, default="", help="Workload name")
    parser.add_argument("-n", "--network_topo", type=str, default="", help="Network topology file")
    parser.add_argument("-c", "--network_conf", type=str, default="", help="Network configuration file")
    return parser.parse_args()


def main():
    args = user_param_parse()
 

if __name__ == "__main__":
    main()
