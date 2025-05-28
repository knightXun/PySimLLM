import os
import ns3
import sys
import time
from dataclasses import dataclass
from collections import defaultdict

# from system.Common import GPUType
import enum
class GPUType(enum.Enum):
    A100 = 0
    A800 = 1
    H100 = 2
    H800 = 3
    NONE = 4


from ns import ns


# NS_LOG_COMPONENT_DEFINE("GENERIC_SIMULATION");

ns.LogComponentEnable("GENERIC_SIMULATION", ns.LOG_LEVEL_INFO)

PGO_TRAINING = False
PATH_TO_PGO_CONFIG = "path_to_pgo_config"

cc_mode = 1
enable_qcn = True
use_dynamic_pfc_threshold = True
packet_payload_size = 1000
l2_chunk_size = 0
l2_ack_interval = 0
pause_time = 5.0
simulator_stop_time = 3.01
data_rate = ""
link_delay = ""
topology_file = ""
flow_file = ""
trace_file = ""
trace_output_file = ""
fct_output_file = "fct.txt"
pfc_output_file = "pfc.txt"
send_output_file = "send.txt"
alpha_resume_interval = 55.0
rp_timer = 0.0
ewma_gain = 1/16
rate_decrease_interval = 4.0
fast_recovery_times = 5
rate_ai = ""
rate_hai = ""
min_rate = "100Mb/s"
dctcp_rate_ai = "1000Mb/s"
clamp_target_rate = False
l2_back_to_zero = False
error_rate_per_link = 0.0
has_win = 1
global_t = 1
mi_thresh = 5
var_win = False
fast_react = True
multi_rate = True
sample_feedback = False
pint_log_base = 1.05
pint_prob = 1.0
u_target = 0.95
int_multi = 1
rate_bound = True
nic_total_pause_time = 0
ack_high_prio = 0
link_down_time = 0
link_down_A = 0
link_down_B = 0
enable_trace = 1
buffer_size = 16
node_num = 0
switch_num = 0
link_num = 0
trace_num = 0
nvswitch_num = 0
gpus_per_server = 0
gpu_type = None  
NVswitchs = []
qp_mon_interval = 100
bw_mon_interval = 10000
qlen_mon_interval = 10000
mon_start = 0
mon_end = 2100000000
qlen_mon_file = ""
bw_mon_file = ""
rate_mon_file = ""
cnp_mon_file = ""
total_flow_file = "/root/astra-sim/extern/network_backend/ns3-interface/simulation/monitor_output/"
total_flow_output = None
rate2kmax = defaultdict(int)
rate2kmin = defaultdict(int)
rate2pmax = defaultdict(float)
topof = None
flowf = None
tracef = None

n = ns.NodeContainer()

nic_rate = 0
maxRtt = 0
maxBdp = 0
serverAddress = []
portNumber = defaultdict(lambda: defaultdict(int))

@dataclass
class Interface:
    idx: int = 0
    up: bool = False
    delay: int = 0
    bw: int = 0

nbr2if = {}
nextHop = {}
pairDelay = {}
pairTxDelay = {}
pairBw = {}
pairBdp = {}
pairRtt = {}

@dataclass
class FlowInput:
    src: int = 0
    dst: int = 0
    pg: int = 0
    maxPacketCount: int = 0
    port: int = 0
    dport: int = 0
    start_time: float = 0.0
    idx: int = 0

flow_input = FlowInput()
flow_num = 0

def node_id_to_ip(id):
    return ns.Ipv4Address(0x0b000001 + ((id // 256) * 0x00010000) + ((id % 256) * 0x00000100))

def ip_to_node_id(ip):
    return (ip.Get() >> 8) & 0xffff

def get_pfc(fout, dev, type):
    fout.write(f"{ns.Simulator.Now().GetTimeStep()} {dev.GetNode().GetId()} {dev.GetNode().GetNodeType()} {dev.GetIfIndex()} {type}\n")

def monitor_qlen(qlen_output, nodes):
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 1:
            sw = node.GetObject(ns.SwitchNode)
            sw.PrintSwitchQlen(qlen_output)
        elif node.GetNodeType() == 2:
            sw = node.GetObject(ns.NVSwitchNode)
            sw.PrintSwitchQlen(qlen_output)
    ns.Simulator.Schedule(ns.MicroSeconds(qlen_mon_interval), monitor_qlen, qlen_output, nodes)

def monitor_bw(bw_output, nodes):
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 1:
            sw = node.GetObject(ns.SwitchNode)
            sw.PrintSwitchBw(bw_output, bw_mon_interval)
        elif node.GetNodeType() == 2:
            sw = node.GetObject(ns.NVSwitchNode)
            sw.PrintSwitchBw(bw_output, bw_mon_interval)
        else:
            host = node
            host.GetObject(ns.RdmaDriver).m_rdma.PrintHostBW(bw_output, bw_mon_interval)
    ns.Simulator.Schedule(ns.MicroSeconds(bw_mon_interval), monitor_bw, bw_output, nodes)

def monitor_qp_rate(rate_output, nodes):
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 0:
            host = node
            host.GetObject(ns.RdmaDriver).m_rdma.PrintQPRate(rate_output)
    ns.Simulator.Schedule(ns.MicroSeconds(qp_mon_interval), monitor_qp_rate, rate_output, nodes)

def monitor_qp_cnp_number(cnp_output, nodes):
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 0:
            host = node
            host.GetObject(ns.RdmaDriver).m_rdma.PrintQPCnpNumber(cnp_output)
    ns.Simulator.Schedule(ns.MicroSeconds(qp_mon_interval), monitor_qp_cnp_number, cnp_output, nodes)

def schedule_monitor():
    with open(qlen_mon_file, "w") as qlen_output:
        qlen_output.write("time, sw_id, port_id, q_id, q_len, port_len\n")
        ns.Simulator.Schedule(ns.MicroSeconds(mon_start), monitor_qlen, qlen_output, n)

    with open(bw_mon_file, "w") as bw_output:
        bw_output.write("time, node_id, port_id, bandwidth\n")
        ns.Simulator.Schedule(ns.MicroSeconds(mon_start), monitor_bw, bw_output, n)

    with open(rate_mon_file, "w") as rate_output:
        rate_output.write("time, src, dst, sport, dport, size, curr_rate\n")
        ns.Simulator.Schedule(ns.MicroSeconds(mon_start), monitor_qp_rate, rate_output, n)

    with open(cnp_mon_file, "w") as cnp_output:
        cnp_output.write("time, src, dst, sport, dport, size, cnp_number\n")
        ns.Simulator.Schedule(ns.MicroSeconds(mon_start), monitor_qp_cnp_number, cnp_output, n)

def CalculateRoute(host):
    q = [host]
    dis = {host: 0}
    delay = {host: 0}
    txDelay = {host: 0}
    bw = {host: 0xffffffffffffffff}
    for i in range(len(q)):
        now = q[i]
        d = dis[now]
        for neighbor, iface in nbr2if[now].items():
            if not iface.up:
                continue
            next_node = neighbor
            if next_node not in dis:
                dis[next_node] = d + 1
                delay[next_node] = delay[now] + iface.delay
                txDelay[next_node] = txDelay[now] + (packet_payload_size * 1000000000 * 8) // iface.bw
                bw[next_node] = min(bw[now], iface.bw)
                if next_node.GetNodeType() in (1, 2):
                    q.append(next_node)
            via_nvswitch = False
            if d + 1 == dis.get(next_node, float('inf')):
                for x in nextHop[next_node][host]:
                    if x.GetNodeType() == 2:
                        via_nvswitch = True
                if not via_nvswitch:
                    if now.GetNodeType() == 2:
                        nextHop[next_node][host].clear()
                    nextHop[next_node][host].append(now)
                elif via_nvswitch and now.GetNodeType() == 2:
                    nextHop[next_node][host].append(now)
                if next_node.GetNodeType() == 0 and not nextHop[next_node][now]:
                    nextHop[next_node][now].append(now)
                    pairBw[next_node.GetId()][now.GetId()] = pairBw[now.GetId()][next_node.GetId()] = iface.bw
    
    for node, dly in delay.items():
        pairDelay[node] = {}
        pairDelay[node][host] = dly
    for node, tx_dly in txDelay.items():
        pairTxDelay[node] = {}
        pairTxDelay[node][host] = tx_dly
    for node, b in bw.items():
        pairBw[node.GetId()] = {}
        pairBw[node.GetId()][host.GetId()] = b

def CalculateRoutes(nodes):
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 0:
            CalculateRoute(node)

def SetRoutingEntries():
    for node, table in nextHop.items():
        for dst, nexts in table.items():
            dstAddr = dst.GetObject(ns.Ipv4).GetAddress(1, 0).GetLocal()
            for next_node in nexts:
                interface = nbr2if[node][next_node].idx
                if node.GetNodeType() == 1:
                    node.GetObject(ns.SwitchNode).AddTableEntry(dstAddr, interface)
                elif node.GetNodeType() == 2:
                    node.GetObject(ns.NVSwitchNode).AddTableEntry(dstAddr, interface)
                    node.GetObject(ns.RdmaDriver).m_rdma.AddTableEntry(dstAddr, interface, True)
                else:
                    is_nvswitch = next_node.GetNodeType() == 2
                    node.GetObject(ns.RdmaDriver).m_rdma.AddTableEntry(dstAddr, interface, is_nvswitch)
                    if next_node.GetId() == dst.GetId():
                        node.GetObject(ns.RdmaDriver).m_rdma.add_nvswitch(dst.GetId())

def printRoutingEntries():
    types = {0: "HOST", 1: "SWITCH", 2: "NVSWITCH"}
    NVSwitch = defaultdict(lambda: defaultdict(list))
    NetSwitch = defaultdict(lambda: defaultdict(list))
    Host = defaultdict(lambda: defaultdict(list))
    
    for src, table in nextHop.items():
        for dst, entries in table.items():
            for next_node in entries:
                interface = nbr2if[src][next_node].idx
                if src.GetNodeType() == 0:
                    Host[src][dst].append((next_node, interface))
                elif src.GetNodeType() == 1:
                    NetSwitch[src][dst].append((next_node, interface))
                elif src.GetNodeType() == 2:
                    NVSwitch[src][dst].append((next_node, interface))

    print("*********************    PRINT SWITCH ROUTING TABLE    *********************")
    for src, table in NetSwitch.items():
        print(f"SWITCH: {src.GetId()}'s routing entries are as follows:")
        for dst, entries in table.items():
            for next_node, interface in entries:
                print(f"To {dst.GetId()}[{types[dst.GetNodeType()]}] via {next_node.GetId()}[{types[next_node.GetNodeType()]}] from port: {interface}")

    print("*********************    PRINT NVSWITCH ROUTING TABLE    *********************")
    for src, table in NVSwitch.items():
        print(f"NVSWITCH: {src.GetId()}'s routing entries are as follows:")
        for dst, entries in table.items():
            for next_node, interface in entries:
                print(f"To {dst.GetId()}[{types[dst.GetNodeType()]}] via {next_node.GetId()}[{types[next_node.GetNodeType()]}] from port: {interface}")

    print("*********************    HOST ROUTING TABLE    *********************")
    for src, table in Host.items():
        print(f"HOST: {src.GetId()}'s routing entries are as follows:")
        for dst, entries in table.items():
            for next_node, interface in entries:
                print(f"To {dst.GetId()}[{types[dst.GetNodeType()]}] via {next_node.GetId()}[{types[next_node.GetNodeType()]}] from port: {interface}")

def validateRoutingEntries():
    return False

def TakeDownLink(nodes, a, b):
    if not nbr2if[a][b].up:
        return
    nbr2if[a][b].up = False
    nbr2if[b][a].up = False
    nextHop.clear()
    CalculateRoutes(nodes)
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 1:
            node.GetObject(ns.SwitchNode).ClearTable()
        elif node.GetNodeType() == 2:
            node.GetObject(ns.NVSwitchNode).ClearTable()
        else:
            node.GetObject(ns.RdmaDriver).m_rdma.ClearTable()
    a.GetDevice(nbr2if[a][b].idx).GetObject(ns.QbbNetDevice).TakeDown()
    b.GetDevice(nbr2if[b][a].idx).GetObject(ns.QbbNetDevice).TakeDown()
    SetRoutingEntries()
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 0:
            node.GetObject(ns.RdmaDriver).m_rdma.RedistributeQp()

def get_output_file_name(config_file, output_file):
    idx = config_file.rfind('/')
    return output_file[:-4] + config_file[idx+7:]

def get_nic_rate(nodes):
    for i in range(nodes.GetN()):
        node = nodes.Get(i)
        if node.GetNodeType() == 0:
            return node.GetDevice(1).GetObject(ns.QbbNetDevice).GetDataRate().GetBitRate()
    return 0

def ReadConf(network_topo, network_conf):
    global topology_file
    topology_file = network_topo
    with open(network_conf, "r") as conf:
        # while True:
        line = conf.readline()
        line = line.strip()
        if not line or line.startswith('#'): pass
            # continue
        parts = line.split()
        key = parts[0]
        values = parts[1:]
        if key == "ENABLE_QCN":
            global enable_qcn
            enable_qcn = int(values[0])
        elif key == "USE_DYNAMIC_PFC_THRESHOLD":
            global use_dynamic_pfc_threshold
            use_dynamic_pfc_threshold = int(values[0])
        elif key == "CLAMP_TARGET_RATE":
            global clamp_target_rate
            clamp_target_rate = int(values[0])
        elif key == "PAUSE_TIME":
            global pause_time
            pause_time = float(values[0])
        elif key == "DATA_RATE":
            global data_rate
            data_rate = ' '.join(values)
        elif key == "LINK_DELAY":
            global link_delay
            link_delay = ' '.join(values)
        elif key == "PACKET_PAYLOAD_SIZE":
            global packet_payload_size
            packet_payload_size = int(values[0])
        elif key == "L2_CHUNK_SIZE":
            global l2_chunk_size
            l2_chunk_size = int(values[0])
        elif key == "L2_ACK_INTERVAL":
            global l2_ack_interval
            l2_ack_interval = int(values[0])
        elif key == "L2_BACK_TO_ZERO":
            global l2_back_to_zero
            l2_back_to_zero = int(values[0])
        elif key == "FLOW_FILE":
            global flow_file
            flow_file = values[0]

        elif key == "TRACE_FILE":
            global trace_file
            trace_file = values[0]
        elif key == "TRACE_OUTPUT_FILE":
            global trace_output_file
            trace_output_file = values[0]
        elif key == "SIMULATOR_STOP_TIME":
            global simulator_stop_time
            simulator_stop_time = float(values[0])
        elif key == "ALPHA_RESUME_INTERVAL":
            global alpha_resume_interval
            alpha_resume_interval = float(values[0])
        elif key == "RP_TIMER":
            global rp_timer
            rp_timer = float(values[0])
        elif key == "EWMA_GAIN":
            global ewma_gain
            ewma_gain = float(values[0])
        elif key == "FAST_RECOVERY_TIMES":
            global fast_recovery_times
            fast_recovery_times = int(values[0])
        elif key == "RATE_AI":
            global rate_ai
            rate_ai = values[0]
        elif key == "RATE_HAI":
            global rate_hai
            rate_hai = values[0]
        elif key == "ERROR_RATE_PER_LINK":
            global error_rate_per_link
            error_rate_per_link = float(values[0])
        elif key == "CC_MODE":
            global cc_mode
            cc_mode = int(values[0])
        elif key == "RATE_DECREASE_INTERVAL":
            global rate_decrease_interval
            rate_decrease_interval = float(values[0])
        elif key == "MIN_RATE":
            global min_rate
            min_rate = values[0]
        elif key == "FCT_OUTPUT_FILE":
            global fct_output_file
            fct_output_file = values[0]
        elif key == "HAS_WIN":
            global has_win
            has_win = int(values[0])
        elif key == "GLOBAL_T":
            global global_t
            global_t = 1  # 强制设为 1（原 C++ 代码行为）
        elif key == "MI_THRESH":
            global mi_thresh
            mi_thresh = int(values[0])
        elif key == "VAR_WIN":
            global var_win
            var_win = int(values[0])
        elif key == "FAST_REACT":
            global fast_react
            fast_react = int(values[0])
        elif key == "U_TARGET":
            global u_target
            u_target = float(values[0])
        elif key == "INT_MULTI":
            global int_multi
            int_multi = int(values[0])
        elif key == "RATE_BOUND":
            global rate_bound
            rate_bound = bool(values[0])
        elif key == "ACK_HIGH_PRIO":
            global ack_high_prio
            ack_high_prio = int(values[0])
        elif key == "DCTCP_RATE_AI":
            global dctcp_rate_ai
            dctcp_rate_ai = values[0]
        elif key == "NIC_TOTAL_PAUSE_TIME":
            global nic_total_pause_time
            nic_total_pause_time = float(values[0])
        elif key == "PFC_OUTPUT_FILE":
            global pfc_output_file
            pfc_output_file = values[0]
        elif key == "LINK_DOWN":
            global link_down_time
            global link_down_A
            global link_down_B
            link_down_time = int(values[0])
            link_down_A = int(values[1])
            link_down_B = int(values[2])
        elif key == "ENABLE_TRACE":
            global enable_trace
            enable_trace = int(values[0])
        elif key == "KMAX_MAP":
            n_k = int(values[0])
            global rate2kmax 
            for i in range(n_k):
                rate2kmax[ int( values[ i*2 + 1 ] ) ] = int( values[ i*2 + 2 ] )
        elif key == "KMIN_MAP":
            n_k = int(values[0]) 
            global rate2kmin 
            for i in range(n_k):
                rate2kmin[ int( values[ i*2 + 1 ] ) ] = int( values[ i*2 + 2 ] )            
        elif key == "PMAX_MAP":
            n_k = int(values[0]) 
            global rate2pmax 
            for i in range(n_k):
                rate2pmax[ int( values[ i*2 + 1 ] ) ] = float( values[ i*2 + 2 ] )   
            
            import pdb; pdb.set_trace()
        elif key == "BUFFER_SIZE":
            global buffer_size
            buffer_size = int(values[0])
        elif key == "QLEN_MON_FILE":
            global qlen_mon_file
            qlen_mon_file = values[0]
            qlen_mon_file = get_output_file_name(network_conf, qlen_mon_file);
        elif key == "BW_MON_FILE":
            global bw_mon_file
            bw_mon_file = values[0]
            bw_mon_file = get_output_file_name(network_conf, bw_mon_file);
        elif key == "RATE_MON_FILE":
            global rate_mon_file
            rate_mon_file = values[0]
            rate_mon_file = get_output_file_name(network_conf, rate_mon_file);
        elif key == "CNP_MON_FILE":
            global cnp_mon_file
            cnp_mon_file = values[0]
            cnp_mon_file = get_output_file_name(network_conf, cnp_mon_file);
        elif key == "MON_START":
            global mon_start
            mon_start = int(values[0])
        elif key == "MON_END":
            global mon_end
            mon_end = int(values[0])
        elif key == "QP_MON_INTERVAL":
            global qp_mon_interval
            qp_mon_interval = int(values[0])
        elif key == "BW_MON_INTERVAL":
            global bw_mon_interval
            bw_mon_interval = int(values[0])
        elif key == "QLEN_MON_INTERVAL":
            global qlen_mon_interval
            qlen_mon_interval = int(values[0])
        elif key == "MULTI_RATE":
            global multi_rate
            multi_rate = ( int(values[0]) == 0 )
        elif key == "SAMPLE_FEEDBACK":
            global sample_feedback
            sample_feedback = ( int(values[0]) == 0 )
        elif key == "PINT_LOG_BASE":
            global pint_log_base
            pint_log_base = float(values[0])
        elif key == "PINT_PROB":
            global pint_prob
            pint_prob = float(values[0])
        else:
            pass

    return True

def SetConfig():
    # 设置 QbbNetDevice 参数
    dynamicth = use_dynamic_pfc_threshold 
    
    ns.Config.SetDefault("ns3::QbbNetDevice::PauseTime", ns.UintegerValue(int(pause_time)))
    ns.Config.SetDefault("ns3::QbbNetDevice::QcnEnabled", ns.BooleanValue(enable_qcn))
    ns.Config.SetDefault("ns3::QbbNetDevice::DynamicThreshold", ns.BooleanValue(dynamicth))
    
    
    # # 设置 IntHop 参数
    ns.IntHop.multi = int_multi
    
    # # 设置 IntHeader 模式
    if cc_mode == 7:
        ns.IntHeader.mode = ns.IntHeader.TS
    elif cc_mode == 3:
        ns.IntHeader.mode = ns.IntHeader.NORMAL
    elif cc_mode == 10:
        ns.IntHeader.mode = ns.IntHeader.PINT
    else:
        ns.IntHeader.mode = ns.IntHeader.NONE
    
    # # PINT 模式特殊处理
    if cc_mode == 10:
        ns.Pint.set_log_base(pint_log_base)
        ns.IntHeader.pint_bytes = ns.Pint.get_n_bytes()
        print(f"PINT bits: {ns.Pint.get_n_bits()} bytes: {ns.Pint.get_n_bytes()}")


def SetupNetwork(qp_finish, send_finish):
    global topof, flowf, tracef, node_num, gpus_per_server, nvswitch_num, switch_num, link_num, gpu_type, flow_num, trace_num
    global topology_file
    global flow_file
    global trace_file
    global serverAddress
    global n

    topof = open(topology_file, "r")
    
    if os.path.exists(flow_file):
        flowf = open(flow_file, "r")
    else:
        flowf = None
        flow_num = 0 
        

    if os.path.exists(trace_file):
        tracef = open(trace_file, "r")
    else:
        tracef = None
        trace_num = 0

    
    node_num, gpus_per_server, nvswitch_num, switch_num, link_num, gpu_type_str = topof.readline().split()
    node_num = int(node_num)
    gpus_per_server = int(gpus_per_server)
    nvswitch_num = int(nvswitch_num)
    switch_num = int(switch_num)
    link_num = int(link_num)
    
    gpu_type = GPUType.NONE
    if gpu_type_str == "A100":
        gpu_type = GPUType.A100
    elif gpu_type_str == "A800":
        gpu_type = GPUType.A800
    elif gpu_type_str == "H100":
        gpu_type = GPUType.H100
    elif gpu_type_str == "H800":
        gpu_type = GPUType.H800
    

    line = topof.readline().strip()
    gpus = list(map(int, line.split()))
    
    
    # import pdb; pdb.set_trace()

    # 创建节点
    node_type = [0] * node_num
    for i in range(nvswitch_num):
        sid = gpus[i] # int(topof.readline())
        node_type[sid] = 2
    for i in range(switch_num):
        sid = gpus[i + nvswitch_num]
        node_type[sid] = 1

    # n = ns.NodeContainer()
    # n.Create(node_num)
    for i in range(node_num):
        if node_type[i] == 0:
            a = ns.Node()
            n.Add(a)
        elif node_type[i] == 1:
            sw = ns.SwitchNode()
            # sw = ns.CreateObject[ns.SwitchNode]()
            # sw.SetAttribute("EcnEnabled", ns.BooleanValue(enable_qcn))
            n.Add(sw)
        elif node_type[i] == 2:
            sw = ns.NVSwitchNode()
            # sw = ns.CreateObject[ns.NVSwitchNode]()
            n.Add(sw)

    # 安装网络协议栈 
    
    import pdb; pdb.set_trace()
    internet = ns.InternetStackHelper()
    internet.Install(n)

    
    for i in range(node_num):
        if n.Get(i).GetNodeType() == 0 or n.Get(i).GetNodeType() == 2:
            serverAddress[i] = node_id_to_ip(i);


    rem = ns.CreateObject[ns.RateErrorModel]( )
    uv = ns.CreateObject[ns.UniformRandomVariable]( )
    rem.SetRandomVariable(uv)
    uv.SetStream(50)
    rem.SetAttribute("ErrorRate", ns.DoubleValue(error_rate_per_link))
    rem.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))

    pfc_file = open(pfc_output_file, 'w')

    qbb = ns.QbbHelper()
    ipv4 = ns.Ipv4AddressHelper()

    for i in range(link_num):
        src, dst, data_rate, link_delay, error_rate = topof.readline().split()

        src = int(src)
        dst = int(dst)
        error_rate = float(error_rate)

        snode = n.Get(src)
        dnode = n.Get(dst)

        # 配置Qbb参数
        qbb.SetDeviceAttribute("DataRate", ns.StringValue(data_rate))
        qbb.SetChannelAttribute("Delay", ns.StringValue(link_delay))

        # 配置错误模型
        if error_rate > 0:
            rem = ns.CreateObject[ns.RateErrorModel]()
            uv = ns.CreateObject[ns.UniformRandomVariable]()
            rem.SetRandomVariable(uv)
            uv.SetStream(50)
            rem.SetAttribute("ErrorRate", ns.DoubleValue(error_rate))
            rem.SetAttribute("ErrorUnit", ns.StringValue("ERROR_UNIT_PACKET"))
            qbb.SetDeviceAttribute("ReceiveErrorModel", ns.PointerValue(rem))
        else:
            qbb.SetDeviceAttribute("ReceiveErrorModel", ns.PointerValue(rem))

        # 安装网络设备
        devices = qbb.Install(snode, dnode)

        # 配置服务器节点接口
        if snode.GetNodeType() == 0 or snode.GetNodeType() == 2:
            ipv4_interface = snode.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv4_interface.AddInterface(devices.Get(0))
            ipv4_interface.AddAddress(
                1, 
                ns.Ipv4InterfaceAddress(serverAddress[src], ns.Ipv4Mask("255.0.0.0"))
            )

        if dnode.GetNodeType() == 0 or dnode.GetNodeType() == 2:
            ipv4_interface = dnode.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv4_interface.AddInterface(devices.Get(1))
            ipv4_interface.AddAddress(
                1, 
                ns.Ipv4InterfaceAddress(serverAddress[dst], ns.Ipv4Mask("255.0.0.0"))
            )

        # 记录接口信息
        qbb_dev0 = ns.qbb.QbbNetDevice.Cast(devices.Get(0))
        qbb_dev1 = ns.qbb.QbbNetDevice.Cast(devices.Get(1))
        
        nbr2if[snode] = {}
        nbr2if[snode][dnode] = Interface(
            qbb_dev0.GetIfIndex(), True, 
            qbb_dev0.GetChannel().GetDelay().GetTimeStep(), 
            qbb_dev0.GetDataRate().GetBitRate()
        )
        
        nbr2if[dnode] = {}
        nbr2if[dnode][snode] = Interface(
            qbb_dev1.GetIfIndex(), True, 
            qbb_dev1.GetChannel().GetDelay().GetTimeStep(), 
            qbb_dev1.GetDataRate().GetBitRate()
        )

        # 配置IP地址
        ipstring = "10.%d.%d.0" % (i//254+1, i%254+1)
        ipv4.SetBase(ns.Ipv4Address(ipstring), ns.Ipv4Mask("255.255.255.0"))
        ipv4.Assign(devices)

        # 连接PFC跟踪
        qbb_dev0.TraceConnectWithoutContext(
            "QbbPfc", 
            ns.core.Callback(lambda p, d: get_pfc(pfc_file, qbb_dev0, p, d)) )
        
        qbb_dev1.TraceConnectWithoutContext(
            "QbbPfc",
            ns.core.Callback(lambda p, d: get_pfc(pfc_file, qbb_dev1, p, d)) )


        nic_rate = get_nic_rate(n)  # 需要实现get_nic_rate函数

        for i in range(node_num):
            node = n.Get(i)
            if node.GetNodeType() == 1:  # 普通交换机
                sw = ns.SwitchNode.Cast(node)
                shift = 3
                
                for j in range(1, sw.GetNDevices()):
                    dev = ns.qbb.QbbNetDevice.Cast(sw.GetDevice(j))
                    rate = dev.GetDataRate().GetBitRate()
                    
                    # 验证参数存在性
                    if rate not in rate2kmin:
                        raise ValueError(f"must set kmin for rate {rate}")
                    if rate not in rate2kmax:
                        raise ValueError(f"must set kmax for rate {rate}") 
                    if rate not in rate2pmax:
                        raise ValueError(f"must set pmax for rate {rate}")
                    
                    # 配置ECN
                    sw.m_mmu.ConfigEcn(j, rate2kmin[rate], rate2kmax[rate], rate2pmax[rate])
                    
                    # 计算headroom
                    channel = ns.qbb.QbbChannel.Cast(dev.GetChannel())
                    delay = channel.GetDelay().GetTimeStep()
                    headroom = rate * delay // 8 // 1000000000 * 3
                    sw.m_mmu.ConfigHdrm(j, headroom)
                    
                    # 配置PFC shift
                    sw.m_mmu.pfc_a_shift[j] = shift
                    current_rate = rate
                    while current_rate > nic_rate and sw.m_mmu.pfc_a_shift[j] > 0:
                        sw.m_mmu.pfc_a_shift[j] -= 1
                        current_rate = current_rate // 2
                        
                # 全局配置
                sw.m_mmu.ConfigNPort(sw.GetNDevices() - 1)
                sw.m_mmu.ConfigBufferSize(buffer_size * 1024 * 1024)
                sw.m_mmu.node_id = sw.GetId()
                
            elif node.GetNodeType() == 2:  # NVSwitch
                sw = ns.NVSwitchNode.Cast(node)
                shift = 3
                
                for j in range(1, sw.GetNDevices()):
                    dev = ns.qbb.QbbNetDevice.Cast(sw.GetDevice(j))
                    rate = dev.GetDataRate().GetBitRate()
                    
                    # 计算headroom
                    channel = ns.qbb.QbbChannel.Cast(dev.GetChannel())
                    delay = channel.GetDelay().GetTimeStep()
                    headroom = rate * delay // 8 // 1000000000 * 3
                    sw.m_mmu.ConfigHdrm(j, headroom)
                    
                    # 配置PFC shift
                    sw.m_mmu.pfc_a_shift[j] = shift
                    current_rate = rate
                    while current_rate > nic_rate and sw.m_mmu.pfc_a_shift[j] > 0:
                        sw.m_mmu.pfc_a_shift[j] -= 1
                        current_rate = current_rate // 2
                        
                # 全局配置
                sw.m_mmu.ConfigNPort(sw.GetNDevices() - 1)
                sw.m_mmu.ConfigBufferSize(buffer_size * 1024 * 1024)
                sw.m_mmu.node_id = sw.GetId()

    fct_output = open(fct_output_file, 'w')
    send_output = open(send_output_file, 'w')

    for i in range(node_num):
        node = n.Get(i)
        if node.GetNodeType() in (0, 2):  # 处理服务器节点
            # 创建RDMA硬件对象
            rdma_hw = ns.CreateObject[ns.RdmaHw]()
            
            # 设置硬件属性
            rdma_hw.SetAttribute("ClampTargetRate", ns.BooleanValue(clamp_target_rate))
            rdma_hw.SetAttribute("AlphaResumInterval", ns.DoubleValue(alpha_resume_interval))
            rdma_hw.SetAttribute("RPTimer", ns.DoubleValue(rp_timer))
            rdma_hw.SetAttribute("FastRecoveryTimes", ns.UintegerValue(fast_recovery_times))
            rdma_hw.SetAttribute("EwmaGain", ns.DoubleValue(ewma_gain))
            rdma_hw.SetAttribute("RateAI", ns.DataRateValue(ns.DataRate(rate_ai)))
            rdma_hw.SetAttribute("RateHAI", ns.DataRateValue(ns.DataRate(rate_hai)))
            rdma_hw.SetAttribute("L2BackToZero", ns.BooleanValue(l2_back_to_zero))
            rdma_hw.SetAttribute("L2ChunkSize", ns.UintegerValue(l2_chunk_size))
            rdma_hw.SetAttribute("L2AckInterval", ns.UintegerValue(l2_ack_interval))
            rdma_hw.SetAttribute("CcMode", ns.UintegerValue(cc_mode))
            rdma_hw.SetAttribute("RateDecreaseInterval", ns.DoubleValue(rate_decrease_interval))
            rdma_hw.SetAttribute("MinRate", ns.DataRateValue(ns.DataRate(min_rate)))
            rdma_hw.SetAttribute("Mtu", ns.UintegerValue(packet_payload_size))
            rdma_hw.SetAttribute("MiThresh", ns.UintegerValue(mi_thresh))
            rdma_hw.SetAttribute("VarWin", ns.BooleanValue(var_win))
            rdma_hw.SetAttribute("FastReact", ns.BooleanValue(fast_react))
            rdma_hw.SetAttribute("MultiRate", ns.BooleanValue(multi_rate))
            rdma_hw.SetAttribute("SampleFeedback", ns.BooleanValue(sample_feedback))
            rdma_hw.SetAttribute("TargetUtil", ns.DoubleValue(u_target))
            rdma_hw.SetAttribute("RateBound", ns.BooleanValue(rate_bound))
            rdma_hw.SetAttribute("DctcpRateAI", ns.DataRateValue(ns.DataRate(dctcp_rate_ai)))
            rdma_hw.SetAttribute("GPUsPerServer", ns.UintegerValue(gpus_per_server))
            rdma_hw.SetPintSmplThresh(pint_prob)
            rdma_hw.SetAttribute("TotalPauseTimes", ns.UintegerValue(nic_total_pause_time))

            # 创建RDMA驱动
            rdma_driver = ns.CreateObject[ns.RdmaDriver]()
            rdma_driver.SetNode(node)
            rdma_driver.SetRdmaHw(rdma_hw)

            # 绑定到节点
            node.AggregateObject(rdma_driver)
            rdma_driver.Init()

            # 连接跟踪信号
            rdma_driver.TraceConnectWithoutContext(
                "QpComplete", 
                lambda qp: qp_finish(fct_output, rdma_driver.GetNetDevice(), qp)
            )
            
            rdma_driver.TraceConnectWithoutContext(
                "SendComplete",
                lambda packet: send_finish(send_output, packet)
            )


    # 配置ACK队列优先级
    if ack_high_prio:
        ns.RdmaEgressQueue.ack_q_idx = 0
    else:
        ns.RdmaEgressQueue.ack_q_idx = 3

    # 计算路由
    CalculateRoutes(n)
    SetRoutingEntries()

    # 计算最大RTT和BDP
    max_rtt = 0
    max_bdp = 0
    for i in range(node_num):
        if n.Get(i).GetNodeType() != 0:
            continue
        for j in range(node_num):
            if n.Get(j).GetNodeType() != 0:
                continue
            # 获取节点对象
            node_i = n.Get(i)
            node_j = n.Get(j)
            
            # 获取延迟参数
            delay = pairDelay[node_i][node_j]
            tx_delay = pairTxDelay[node_i][node_j]
            rtt = delay * 2 + tx_delay
            bw = pairBw[i][j]
            
            # 计算BDP
            bdp = rtt * bw // 1000000000 // 8
            pairBdp[node_i] = {}
            pairBdp[node_i][node_j] = bdp
            pairRtt[i][j] = rtt
            
            if bdp > max_bdp:
                max_bdp = bdp
            if rtt > max_rtt:
                max_rtt = rtt

    print(f"maxRtt={max_rtt} maxBdp={max_bdp}")

    # 配置交换机参数
    for i in range(node_num):
        node = n.Get(i)
        if node.GetNodeType() == 1:  # 交换机节点
            sw = ns.SwitchNode.Cast(node)
            sw.SetAttribute("CcMode", ns.UintegerValue(cc_mode))
            sw.SetAttribute("MaxRtt", ns.UintegerValue(max_rtt))

    # 处理跟踪节点
    trace_nodes = ns.NodeContainer()
    for _ in range(trace_num):
        nid = int(tracef.readline())
        if nid < n.GetN():
            trace_nodes = ns.NodeContainer(trace_nodes, n.Get(nid));

    # 启用跟踪
    trace_output = open(trace_output_file, 'w') if enable_trace else None
    if enable_trace:
        qbb.EnableTracing(trace_output, trace_nodes)

    sim_setting = ns.SimSetting()
    for node, interfaces in nbr2if.items():
        for neighbor, info in interfaces.items():
            dev = ns.qbb.QbbNetDevice.Cast(node.GetDevice(info.idx))
            sim_setting.port_speed[node.GetId()][info.idx] = dev.GetDataRate().GetBitRate()
    sim_setting.win = max_bdp
    if trace_output:
        sim_setting.Serialize(trace_output)

    # 创建应用程序
    ns.LogComponent.Enable("Application", ns.core.LOG_LEVEL_INFO)
    inter_packet_interval = ns.Seconds(0.0000005 / 2)

    # 初始化端口号矩阵
    port_number = defaultdict(lambda: defaultdict(int))
    for i in range(node_num):
        node = n.Get(i)
        if node.GetNodeType() in (0, 2):
            for j in range(node_num):
                peer = n.Get(j)
                if peer.GetNodeType() in (0, 2):
                    port_number[i][j] = 10000

    # 关闭输入文件
    topof.close()
    tracef.close()

    if link_down_time > 0:
        ns.Simulator.Schedule(
            ns.Seconds(2) + ns.MicroSeconds(link_down_time),
            TakeDownLink,
            n,
            n.Get(link_down_A),
            n.Get(link_down_B)
        )


    # ns.Simulator.Stop(ns.Seconds(simulator_stop_time))
    # ns.Simulator.Run()
    # ns.Simulator.Destroy()

if __name__ == "__main__":
    import time 
    begint = time.time_ns()

    ReadConf("Spectrum-X_128g_8gps_100Gbps_A100", "SimAI.conf")

    print("Read Conf Done.")


    SetConfig()
    SetupNetwork(None, None)  # 需要实现具体的qp_finish和send_finish回调
    
    print("Running Simulation.")


    endt = time.time_ns()
    
    ns.Simulator.Stop(ns.Seconds(100))
    ns.Simulator.Run()
    ns.Simulator.Destroy()