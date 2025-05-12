import sys
import time
import os
import threading
from collections import defaultdict
import ns3

# 模拟AstraSim命名空间
class AstraSim:
    class ncclFlowTag:
        def __init__(self):
            self.current_flow_id = -1
            self.child_flow_id = -1
            self.tag_id = -1
            self.sender_node = -1
            self.receiver_node = -1
            self.channel_id = -1
            self.nvls_on = False

    class RecvPacketEventHadndlerData:
        def __init__(self):
            self.flowTag = AstraSim.ncclFlowTag()

    class SendPacketEventHandlerData:
        def __init__(self):
            self.flowTag = AstraSim.ncclFlowTag()

    class Sys:
        @staticmethod
        def boostedTick():
            return ns3.Simulator.Now().GetNanoSeconds()  # 使用NS3模拟器时间

# 日志单例类
class MockNcclLog:
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._logger = ns3.LogComponentGet("MockNcclLog")
            ns3.LogComponentEnable("MockNcclLog", ns3.LOG_LEVEL_ALL)
        return cls._instance

    def writeLog(self, level, message, *args):
        log_level_map = {
            "DEBUG": ns3.LOG_DEBUG,
            "INFO": ns3.LOG_INFO,
            "ERROR": ns3.LOG_ERROR
        }
        self._logger.Log(log_level_map[level], message % args)

# 全局状态管理
class GlobalState:
    portNumber = defaultdict(lambda: defaultdict(int))  # portNumber[src][dst]
    serverAddress = {}  # 节点IP地址映射
    pairRtt = defaultdict(lambda: defaultdict(int))  # 节点间RTT
    pairBw = defaultdict(lambda: defaultdict(int))  # 节点间带宽
    maxBdp = 0
    pairBdp = defaultdict(lambda: defaultdict(int))
    has_win = False
    global_t = 0

# 核心数据结构
receiver_pending_queue = defaultdict(AstraSim.ncclFlowTag)
sender_src_port_map = defaultdict(AstraSim.ncclFlowTag)
expeRecvHash = defaultdict()
recvHash = defaultdict(int)
sentHash = defaultdict()
nodeHash = defaultdict(int)
waiting_to_sent_callback = defaultdict(int)
waiting_to_notify_receiver = defaultdict(int)
received_chunksize = defaultdict(int)
sent_chunksize = defaultdict(int)

# 核心函数实现
def is_sending_finished(src, dst, flow_tag):
    key = (flow_tag.current_flow_id, (src, dst))
    if key in waiting_to_sent_callback:
        waiting_to_sent_callback[key] -= 1
        if waiting_to_sent_callback[key] == 0:
            del waiting_to_sent_callback[key]
            return True
    return False

def is_receive_finished(src, dst, flow_tag):
    key = (flow_tag.current_flow_id, (src, dst))
    nccl_log = MockNcclLog()
    if key in waiting_to_notify_receiver:
        nccl_log.writeLog("DEBUG", "is_receive_finished waiting_to_notify_receiver tag_id %d src %d dst %d count %d",
                         flow_tag.current_flow_id, src, dst, waiting_to_notify_receiver[key])
        waiting_to_notify_receiver[key] -= 1
        if waiting_to_notify_receiver[key] == 0:
            del waiting_to_notify_receiver[key]
            return True
    return False

def send_flow(src, dst, max_packet_count, msg_handler, fun_arg, tag, request):
    nccl_log = MockNcclLog()
    qps_per_connection = 1  # 对应原宏定义_QPS_PER_CONNECTION_
    packet_count = (max_packet_count + qps_per_connection - 1) // qps_per_connection
    left_packet_count = max_packet_count

    for index in range(qps_per_connection):
        real_packet_count = min(packet_count, left_packet_count)
        left_packet_count -= real_packet_count
        port = GlobalState.portNumber[src][dst]
        GlobalState.portNumber[src][dst] += 1  # 模拟端口自增

        # 临界区处理（NS3 MTP模式）
        if os.getenv("NS3_MTP"):
            with threading.Lock():
                sender_src_port_map[(port, (src, dst))] = request.flowTag
        else:
            sender_src_port_map[(port, (src, dst))] = request.flowTag

        flow_id = request.flowTag.current_flow_id
        nvls_on = request.flowTag.nvls_on
        send_lat = 6000  # 默认发送延迟（ns）
        send_lat_env = os.getenv("AS_SEND_LAT")
        if send_lat_env:
            try:
                send_lat = int(send_lat_env) * 1000  # 转换为纳秒
            except ValueError:
                nccl_log.writeLog("ERROR", "send_lat set error")
                sys.exit(-1)

        # 创建NS3 RDMA客户端应用
        pg = 3
        dport = 100
        src_ip = GlobalState.serverAddress[src]
        dst_ip = GlobalState.serverAddress[dst]
        
        client_helper = ns3.RdmaClientHelper(
            pg, src_ip, dst_ip, port, dport, real_packet_count,
            GlobalState.maxBdp if GlobalState.has_win and GlobalState.global_t == 1 
            else GlobalState.pairBdp[src][dst],
            GlobalState.maxRtt if GlobalState.global_t == 1 
            else GlobalState.pairRtt[src][dst],
            msg_handler, fun_arg, tag, src, dst
        )

        if nvls_on:
            client_helper.SetAttribute("NVLS_enable", ns3.UintegerValue(1))

        # 安装应用并启动
        app_container = client_helper.Install(ns3.NodeList.GetNode(src))
        app_container.Start(ns3.Time(ns3.NanoSeconds(send_lat)))

        # 更新等待计数器
        key = (request.flowTag.current_flow_id, (src, dst))
        waiting_to_sent_callback[key] += 1
        waiting_to_notify_receiver[key] += 1

        nccl_log.writeLog("DEBUG", "waiting_to_notify_receiver current_flow_id %d src %d dst %d count %d",
                         request.flowTag.current_flow_id, src, dst, waiting_to_notify_receiver[key])

def notify_receiver_receive_data(sender_node, receiver_node, message_size, flow_tag):
    nccl_log = MockNcclLog()
    tag = flow_tag.tag_id
    key = (tag, (sender_node, receiver_node))

    # 临界区处理
    if os.getenv("NS3_MTP"):
        with threading.Lock():
            _notify_receiver_core(key, message_size, flow_tag, nccl_log)
    else:
        _notify_receiver_core(key, message_size, flow_tag, nccl_log)

    # 更新节点哈希
    node_hash_key = (receiver_node, 1)
    nodeHash[node_hash_key] = nodeHash.get(node_hash_key, 0) + message_size

def _notify_receiver_core(key, message_size, flow_tag, nccl_log):
    if key in expeRecvHash:
        t2 = expeRecvHash[key]
        ehd = t2.fun_arg  # 假设为RecvPacketEventHadndlerData实例
        nccl_log.writeLog("DEBUG", "%d notify recevier: %d message size: %d t2.count: %d channle id: %d",
                         flow_tag.sender_node, flow_tag.receiver_node, message_size, t2.count, flow_tag.channel_id)
        
        if message_size == t2.count:
            del expeRecvHash[key]
            ehd.flowTag = flow_tag
            t2.msg_handler(t2.fun_arg)
        elif message_size > t2.count:
            recvHash[key] = message_size - t2.count
            del expeRecvHash[key]
            ehd.flowTag = flow_tag
            t2.msg_handler(t2.fun_arg)
        else:
            t2.count -= message_size
            expeRecvHash[key] = t2
    else:
        receiver_pending_queue[((receiver_node, sender_node), flow_tag.tag_id)] = flow_tag
        recvHash[key] = recvHash.get(key, 0) + message_size

def qp_finish(fout, q):
    nccl_log = MockNcclLog()
    sid = ns3.ip_to_node_id(q.GetSip())  # 假设已实现ip_to_node_id绑定
    did = ns3.ip_to_node_id(q.GetDip())
    port = q.GetSport()

    # 获取flowTag
    key = (port, (sid, did))
    if key not in sender_src_port_map:
        nccl_log.writeLog("ERROR", "could not find the tag, there must be something wrong")
        sys.exit(-1)
    flow_tag = sender_src_port_map[key]
    del sender_src_port_map[key]

    # 更新接收分片计数
    chunksize_key = (flow_tag.current_flow_id, (sid, did))
    received_chunksize[chunksize_key] += q.GetSize()

    if not is_receive_finished(sid, did, flow_tag):
        return

    # 通知接收完成
    notify_size = received_chunksize.pop(chunksize_key)
    notify_receiver_receive_data(sid, did, notify_size, flow_tag)

    # 清理NS3资源
    dst_node = ns3.NodeList.GetNode(did)
    rdma = dst_node.GetObject(ns3.RdmaDriver.GetTypeId())
    rdma.GetRdma().DeleteRxQp(q.GetSip(), q.GetPg(), port)

def send_finish(fout, q):
    sid = ns3.ip_to_node_id(q.GetSip())
    did = ns3.ip_to_node_id(q.GetDip())
    port = q.GetSport()

    # 获取flowTag
    key = (port, (sid, did))
    flow_tag = sender_src_port_map[key]

    # 更新发送分片计数
    chunksize_key = (flow_tag.current_flow_id, (sid, did))
    sent_chunksize[chunksize_key] += q.GetSize()

    if not is_sending_finished(sid, did, flow_tag):
        return

    # 通知发送完成
    all_sent = sent_chunksize.pop(chunksize_key)
    notify_sender_sending_finished(sid, did, all_sent, flow_tag)

def setup_network(qp_finish_cb, send_finish_cb):
    # 示例拓扑：2个终端节点 + 1个交换机
    nodes = ns3.NodeContainer()
    nodes.Create(3)  # node0: sender, node1: receiver, node2: switch

    # 配置点到点链路
    p2p = ns3.PointToPointHelper()
    p2p.SetDeviceAttribute("DataRate", ns3.StringValue("10Gbps"))
    p2p.SetChannelAttribute("Delay", ns3.StringValue("2us"))

    # 连接节点到交换机
    sender_dev = p2p.Install(nodes.Get(0), nodes.Get(2))
    receiver_dev = p2p.Install(nodes.Get(1), nodes.Get(2))

    # 安装协议栈
    internet = ns3.InternetStackHelper()
    internet.Install(nodes)

    # 分配IP地址
    ipv4 = ns3.Ipv4AddressHelper()
    ipv4.SetBase("10.1.1.0", "255.255.255.0")
    sender_if = ipv4.Assign(sender_dev)
    receiver_if = ipv4.Assign(receiver_dev)

    # 记录全局IP映射
    GlobalState.serverAddress[0] = sender_if.GetAddress(0)
    GlobalState.serverAddress[1] = receiver_if.GetAddress(0)

    # 配置路由
    ns3.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

    # 注册完成回调（需要将C++回调绑定到Python函数）
    ns3.RdmaHelper.SetQpFinishCallback(qp_finish_cb)
    ns3.RdmaHelper.SetSendFinishCallback(send_finish_cb)

def main1(network_topo, network_conf):
    # 初始化NS3
    ns3.LogComponentEnable("RdmaClient", ns3.LOG_LEVEL_ALL)
    ns3.Simulator.Stop(ns3.Seconds(10.0))

    # 读取配置（示例简化）
    GlobalState.pairRtt[0][1] = 1000000  # 1ms RTT（ns）
    GlobalState.pairBw[0][1] = 10e9  # 10Gbps
    GlobalState.maxBdp = 1000000  # 示例BDP值

    # 搭建网络
    setup_network(qp_finish, send_finish)

    # 示例发送请求
    request = AstraSim.SendPacketEventHandlerData()
    request.flowTag.current_flow_id = 1
    request.flowTag.sender_node = 0
    request.flowTag.receiver_node = 1
    request.flowTag.nvls_on = False

    # 启动发送
    send_flow(0, 1, 1000, lambda x: print("Send callback"), request, 0, request)

    # 运行模拟
    ns3.Simulator.Run()
    ns3.Simulator.Destroy()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ns3_nccl_simulation.py <network_topo> <network_conf>")
        sys.exit(1)
    main1(sys.argv[1], sys.argv[2])
