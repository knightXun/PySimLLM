import re
import sys
from enum import Enum

from Common import GPUType

BUSBW_PATH = ""

class ModeType(Enum):
    NONE = 0
    ASTRA_SIM = 1
    MOCKNCCL = 2
    ANALYTICAL = 3


class NetWorkParam:
    def __init__(self):
        self.node_num = 0
        self.switch_num = 0
        self.link_num = 0
        self.trace_num = 0
        self.nvswitch_num = 0
        self.gpus_per_server = 0
        self.nics_per_server = 0
        self.nvlink_bw = 0
        self.nic_bw = 0
        self.gpu_type = GPUType()
        self.tp_ar = -1.0
        self.tp_ag = -1.0
        self.tp_rs = -1.0
        self.tp_ata = -1.0
        self.dp_ar = -1.0
        self.dp_ag = -1.0
        self.dp_rs = -1.0
        self.dp_ata = -1.0
        self.ep_ar = -1.0
        self.ep_ag = -1.0
        self.ep_rs = -1.0
        self.ep_ata = -1.0
        self.pp = -1.0
        self.dp_overlap_ratio = 0
        self.tp_overlap_ratio = 0
        self.ep_overlap_ratio = 0
        self.pp_overlap_ratio = 1
        self.NVswitchs = []
        self.all_gpus = []
        self.visual = 0

class UserParam:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.thread = 1
            cls._instance.gpus = []
            cls._instance.workload = ""
            cls._instance.res = "None"
            cls._instance.comm_scale = 1
            cls._instance.mode = ModeType.MOCKNCCL
            cls._instance.net_work_param = NetWorkParam()
        return cls._instance

    def parseYaml(self, params, filename):
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
                currentSection = ""
                for line in lines[1:]:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.endswith(':'):
                        currentSection = line[:-1]
                    else:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().split(',')[0]
                            valueStr = parts[1].strip()
                            if valueStr != "null":
                                value = float(valueStr)
                                if currentSection == "TP":
                                    if key == "allreduce":
                                        params.tp_ar = value
                                    elif key == "allgather":
                                        params.tp_ag = value
                                    elif key == "reducescatter":
                                        params.tp_rs = value
                                    elif key == "alltoall":
                                        params.tp_ata = value
                                elif currentSection == "DP":
                                    if key == "allreduce":
                                        params.dp_ar = value
                                    elif key == "allgather":
                                        params.dp_ag = value
                                    elif key == "reducescatter":
                                        params.dp_rs = value
                                    elif key == "alltoall":
                                        params.dp_ata = value
                                elif currentSection == "EP":
                                    if key == "allreduce":
                                        params.ep_ar = value
                                    elif key == "allgather":
                                        params.ep_ag = value
                                    elif key == "reducescatter":
                                        params.ep_rs = value
                                    elif key == "alltoall":
                                        params.ep_ata = value
                                elif currentSection == "PP":
                                    if key == "busbw":
                                        params.pp = value
        except FileNotFoundError:
            print(f"Unable to open file: {filename}")
            sys.exit(-1)

    def printHelp(self):
        print(" ____  _              _    ___        _                _       _   _           _ ")
        print("/ ___|(_)_ __ ___    / \\  |_ _|      / \\   _ __   __ _| |_   _| |_(_) ___ __ _| |")
        print("\\___ \\| | '_ ' _ \\  / _ \\  | |_____ / _ \\ | '_ \\ / _' | | | | | __| |/ __/ _' | |")
        print(" ___) | | | | | | |/ ___ \\ | |_____/ ___ \\| | | | (_| | | |_| | |_| | (_| (_| | |")
        print("|____/|_|_| |_| |_/_/   \\_\\___|   /_/   \\_\\_| |_|\\__,_|_|\\__, |\\__|_|\\___\\__,_|_|")
        print("                                                           |___/                   ")
        print("-w,       --workload            Workloads, must set")
        print("-g,       --gpus                Number of GPUs, default 1")
        print("-g_p_s,   --gpus-per-server     GPUs per server")
        print("-r,       --result              Output results path, default: ./results/")
        print("-busbw,   --bus-bandwidth       Bus bandwidth file, must set")
        print("-v,       --visual              Enable visual output (Default disable)")
        print("-dp_o,    --dp-overlap-ratio    DP overlap ratio [float: 0.0-1.0] (Default: 0.0)")
        print("-ep_o,    --ep-overlap-ratio    EP overlap ratio [float: 0.0-1.0] (Default: 0.0)")
        print("-tp_o,    --tp-overlap-ratio    TP overlap ratio [float: 0.0-1.0] (Default: 0.0)")
        print("-pp_o,    --pp-overlap-ratio    PP overlap ratio [float: 0.0-1.0] (Default: 1.0)")

    def printError(self, arg):
        print(f"Error: Missing value for argument '{arg}'.")
        return 1

    def printUnknownOption(self, arg):
        print(f"Error: Unknown option '{arg}'.")
        return 1

    def parseArg(self, argc, argv):
        i = 1
        while i < argc:
            arg = argv[i]
            if arg in ["-h", "--help"]:
                self.printHelp()
                return 1
            elif arg in ["-w", "--workload"]:
                i += 1
                if i < argc:
                    self.workload = argv[i]
                else:
                    return self.printError(arg)
            elif arg in ["-g", "--gpus"]:
                i += 1
                if i < argc:
                    self.gpus.append(int(argv[i]))
                else:
                    return self.printError(arg)
            elif arg in ["-r", "--result"]:
                i += 1
                if i < argc:
                    self.res = argv[i]
                else:
                    return self.printError(arg)
            elif arg in ["-g_p_s", "--gpus-per-server"]:
                i += 1
                if i < argc:
                    self.net_work_param.gpus_per_server = int(argv[i])
                else:
                    return self.printError(arg)
            elif arg in ["-busbw", "--bus-bandwidth"]:
                i += 1
                if i < argc:
                    self.parseYaml(self.net_work_param, argv[i])
                else:
                    return self.printError(arg)
            elif arg in ["--dp-overlap-ratio", "-dp_o"]:
                i += 1
                if i < argc:
                    self.net_work_param.dp_overlap_ratio = float(argv[i])
                else:
                    return self.printError(arg)
            elif arg in ["--tp-overlap-ratio", "-tp_o"]:
                i += 1
                if i < argc:
                    self.net_work_param.tp_overlap_ratio = float(argv[i])
                else:
                    return self.printError(arg)
            elif arg in ["--ep-overlap-ratio", "-ep_o"]:
                i += 1
                if i < argc:
                    self.net_work_param.ep_overlap_ratio = float(argv[i])
                else:
                    return self.printError(arg)
            elif arg in ["--pp-overlap-ratio", "-pp_o"]:
                i += 1
                if i < argc:
                    self.net_work_param.pp_overlap_ratio = float(argv[i])
                else:
                    return self.printError(arg)
            elif arg in ["-v", "--visual"]:
                self.net_work_param.visual = 1
            else:
                return self.printUnknownOption(arg)
            i += 1

        if self.gpus:
            self.net_work_param.nvswitch_num = self.gpus[0] // self.net_work_param.gpus_per_server
            self.net_work_param.switch_num = 120 + self.net_work_param.gpus_per_server
            self.net_work_param.node_num = self.net_work_param.nvswitch_num + self.net_work_param.switch_num + self.gpus[0]

        if self.res == "None" or self.res.endswith('/'):
            full_path = self.workload
            model_info = full_path
            last_slash_pos = full_path.rfind('/')
            if last_slash_pos != -1:
                model_info = full_path[last_slash_pos + 1:]

            model_name = ""
            world_size = 0
            tp = 0
            pp = 0
            ep = 0
            gbs = 0
            mbs = 0
            seq = 0

            world_size_pos = model_info.find("world_size")
            if world_size_pos != -1:
                model_name = model_info[:world_size_pos - 1]

            param_regex = re.compile(r'(world_size|tp|pp|ep|gbs|mbs|seq)(\d+)')
            for match in param_regex.finditer(model_info):
                param_name = match.group(1)
                param_value = int(match.group(2))

                if param_name == "world_size":
                    world_size = param_value
                elif param_name == "tp":
                    tp = param_value
                elif param_name == "pp":
                    pp = param_value
                elif param_name == "ep":
                    ep = param_value
                elif param_name == "gbs":
                    gbs = param_value
                elif param_name == "mbs":
                    mbs = param_value
                elif param_name == "seq":
                    seq = param_value

            dp = world_size // (tp * pp)
            ga = gbs / (dp * mbs)

            result = f"{model_name}-tp{tp}-pp{pp}-dp{dp}-ga{int(ga)}-ep{ep}-NVL{self.net_work_param.gpus_per_server}-DP{self.net_work_param.dp_overlap_ratio}-"
            if self.res.endswith('/'):
                self.res = self.res + result
            else:
                self.res = result

        return 0