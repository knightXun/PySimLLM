#TODO: 增加新的
import math
import time
from enum import Enum
from threading import Condition, Lock, RLock
from typing import Dict, List, Tuple, Optional, Any
import heapq
from datetime import datetime
from collections import defaultdict, deque


from MemBus import MemBus
from MyPacket import MyPacket
from Algorithm import Algorithm
from MockNcclLog import MockNcclLog, NcclLogLevel
from Sys import Sys
from StreamStat import StreamStat
from Common import ComType, EventType, StreamState

from system.topology.RingTopology import RingTopology
from AstraNetworkAPI import sim_request, req_type_e
from system.MockNcclChannel import *
from system.SendPacketEventHandlerData import SendPacketEventHandlerData
from PacketBundle import PacketBundle
from system.RecvPacketEventHandlerData import RecvPacketEventHandlerData

# 主要类实现
class NcclTreeFlowModel(Algorithm):
    g_flow_inCriticalSection = False
    _flow_in_critical_lock = Lock()

    class FlowCriticalSection:
        def __enter__(self):
            while NcclTreeFlowModel.g_flow_inCriticalSection:
                pass
            NcclTreeFlowModel.g_flow_inCriticalSection = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            NcclTreeFlowModel.g_flow_inCriticalSection = False

    def __init__(self, 
                com_type: ComType,
                node_id: int,
                layer_num: int,
                ring_topology: RingTopology,
                data_size: int,
                direction: Any, 
                injection_policy: Any,
                boost_mode: bool,
                ptr_flow_models: Optional[Dict[Tuple[int, int], SingleFlow]],
                tree_channels: int):
        super().__init__(layer_num)
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.comType = com_type
        self.id = node_id
        self.logicalTopology = ring_topology
        self.data_size = data_size
        self.nodes_in_ring = ring_topology.get_nodes_in_ring()
        self.parallel_reduce = 1
        self.toggle = False
        self.name = "Ring"
        self.enabled = True
        self.m_channels = tree_channels
        self.judge_exit_flag = False
        self.send_packets = 0
        self.recv_packets = 0
        self.pQps = {}
        self.zero_latency_packets = defaultdict(int)
        self.non_zero_latency_packets = defaultdict(int)
        self._flow_models: Dict[Tuple[int, int], SingleFlow] = {}
        self.free_packets = defaultdict(int)
        self.indegree_mapping = defaultdict(int)
        self.packets = defaultdict(deque)
        self.transmition = MemBus.Transmition.Fast if ring_topology.dimension == RingTopology.Dimension.Local else MemBus.Transmition.Usual

        if boost_mode:
            self.enabled = ring_topology.is_enabled()  # 需要实现is_enabled方法

        if ptr_flow_models:
            for key, flow in ptr_flow_models.items():
                if flow.dest == self.id:
                    self.free_packets[(flow.channel_id, flow.src)] += 1
                    self._flow_models[key] = flow
                    self.recv_packets += 1
                if flow.src == self.id:
                    self._stream_count[flow.channel_id] += 1
                    self._flow_models[key] = flow
                    self.send_packets += 1

        for channel_id in range(self.m_channels):
            self.zero_latency_packets[channel_id] = 0
            self.non_zero_latency_packets[channel_id] = 0

        self.init_indegree_mapping()

        # 根据通信类型设置最终数据大小
        if com_type == ComType.All_Reduce:
            self.final_data_size = data_size
        elif com_type == ComType.All_Gather:
            self.final_data_size = data_size * self.nodes_in_ring
        elif com_type == ComType.Reduce_Scatter:
            self.final_data_size = data_size // self.nodes_in_ring
        elif com_type == ComType.All_to_All:
            self.final_data_size = data_size

    def init_indegree_mapping(self):
        for (channel_id, flow_id), flow in self._flow_models.items():
            if flow.src != self.id:
                continue
            self.indegree_mapping[flow_id] = len(flow.parent_flow_id)

    def get_non_zero_latency_packets(self):
        return (self.nodes_in_ring - 1) * self.parallel_reduce * 1


    def run(self, event, data):
        """处理各种事件类型的主函数"""
        ehd = data  # 假设已进行类型转换
        NcclLog = MockNcclLog.get_instance()

        if event == EventType.General:
            # 处理通用事件
            channel_id = ehd.channel_id
            flow_id = ehd.flow_id
            
            if not self.PHY_MTP:
                self.ready(channel_id, flow_id)
            else:
                self.phy_ready(channel_id, flow_id)
        
        elif event == EventType.PacketReceived:
            # 处理数据包接收事件
            rcehd = ehd  # 假设已进行类型转换
            flowTag = rcehd.flowTag
            received_flow_id = flowTag.current_flow_id
            channel_id = flowTag.channel_id
            next_flow_list = flowTag.tree_flow_list
            
            if self.PHY_MTP:
                # PHY_MTP模式处理
                self.recv_packets -= 1
                if not self.phy_iteratable(channel_id):
                    return
            else:
                # 非PHY_MTP模式处理
                flow_exist = False if next_flow_list else True
                for next_flow_id in next_flow_list:
                    if next_flow_id == -1 or (channel_id, next_flow_id) in self._flow_models:
                        flow_exist = True
                    else:
                        flow_exist = False
                        break
                
                assert flow_exist is True
                
                # 检查流是否完成
                cs = self.FlowCriticalSection()
                key = (channel_id, flowTag.sender_node)
                self.free_packets[key] -= 1
                
                tag = True
                for i in range(self.m_channels):
                    if self._stream_count[i] != 0:
                        tag = False
                        break
                
                cs.exit_section()
                
                if tag:
                    self.ready(channel_id, -1)
                    self.iteratable(channel_id)
                    return
            
            # 记录接收日志
            key = (channel_id, flowTag.sender_node)
            NcclLog.write_log(
                NcclLogLevel.DEBUG,
                "PacketReceived sender_node: %d receiver %d current_flow id: %d channel_id: %d tag_id %d free_packets %d next_flow_list.size %d",
                flowTag.sender_node, flowTag.receiver_node, flowTag.current_flow_id,
                flowTag.channel_id, flowTag.tag_id, self.free_packets.get(key, 0),
                len(next_flow_list)
            )
            
            if self.PHY_MTP:
                # PHY_MTP模式下处理下一个流
                for next_flow_id in next_flow_list:
                    self.indegree_mapping[next_flow_id] -= 1
                    if self.indegree_mapping[next_flow_id] == 0:
                        self.phy_ready(channel_id, next_flow_id)
            else:
                # 非PHY_MTP模式下处理下一个流
                flow_exist = True
                flow_send = False
                recv_finished_tag = True
                
                # 检查所有free_packets是否为0
                for value in self.free_packets.values():
                    if value != 0:
                        recv_finished_tag = False
                        break
                
                NcclLog.write_log(NcclLogLevel.DEBUG, "next_flow_list.size %d", len(next_flow_list))
                
                for next_flow_id in next_flow_list:
                    cs = self.FlowCriticalSection()
                    
                    if next_flow_id not in self.indegree_mapping:
                        flow_exist = False
                        cs.exit_section()
                        break
                    
                    self.indegree_mapping[next_flow_id] -= 1
                    if self.indegree_mapping[next_flow_id] == 0:
                        cur_flow = self._flow_models[(channel_id, next_flow_id)]
                        cs.exit_section()
                        self.insert_packets(channel_id, next_flow_id)
                    else:
                        cs.exit_section()
                
                assert flow_exist is True
        
        elif event == EventType.StreamInit:
            # 处理流初始化事件
            if self.PHY_MTP:
                # PHY_MTP模式下的流初始化
                import mpi4py.MPI as MPI
                MPI.COMM_WORLD.Barrier()
                
                for _, single_flow in self._flow_models.items():
                    if single_flow.src == self.id or single_flow.dest == self.id:
                        if self.PHY_RDMA:
                            self.flow_rdma.ibv_create_peer_qp(
                                self.id, single_flow.channel_id, single_flow.src,
                                single_flow.dest, single_flow.chunk_count + 1,
                                single_flow.chunk_id, single_flow.flow_size
                            )
                
                MPI.COMM_WORLD.Barrier()
                now_us = int(time.time() * 1e6)
                
                NcclLog.write_log(NcclLogLevel.DEBUG, "streamInit time %lld", now_us)
                
                self.start_time = time.time()
            
            # 初始化接收准备状态
            for i in range(self.parallel_reduce):
                if not self.PHY_MTP:
                    self.init_recv_ready()
                
                for j in range(self.m_channels):
                    for _, flow_model in self._flow_models.items():
                        if flow_model.src != self.id:
                            continue
                        
                        parent_list = flow_model.parent_flow_id
                        if (not parent_list) and flow_model.channel_id == j:
                            if self.PHY_MTP:
                                if flow_model.chunk_id == 0:
                                    self.phy_ready(j, flow_model.flow_id)
                            else:
                                if flow_model.chunk_id == 0:
                                    key = (
                                        flow_model.channel_id,
                                        (flow_model.src, flow_model.dest)
                                    )
                                    self.pQps.peer_qps[key] = 0
                                    self.insert_packets(j, flow_model.flow_id)
                                else:
                                    key = (
                                        flow_model.channel_id,
                                        (flow_model.src, flow_model.dest)
                                    )
                                    self.pQps.peer_wating_tasks[key].append(flow_model.flow_id)
                
                if self.PHY_MTP:
                    self.waiting_to_exit()
                    NcclLog.write_log(NcclLogLevel.DEBUG, "NcclTreeFlowModel::waiting_to_exit end")
        
        elif event == EventType.PacketSentFinshed:
            rcehd = ehd  # 假设已进行类型转换
            flowTag = rcehd.flowTag
            sent_flow_id = flowTag.current_flow_id
            channel_id = flowTag.channel_id
            next_flow_list = flowTag.tree_flow_list
            
            NcclLog.write_log(NcclLogLevel.DEBUG,
                "PacketSentFinshed src %d dst %d channel_id %d flow_id %d",
                flowTag.sender_node, flowTag.receiver_node,
                flowTag.channel_id, flowTag.current_flow_id
            )
            
            self.reduce(channel_id, sent_flow_id)
            flow_exist = False if next_flow_list else True
            
            if not self.PHY_MTP:
                key = (flowTag.channel_id, (flowTag.sender_node, flowTag.receiver_node))
                
                cs = self.FlowCriticalSection()
                self.pQps.peer_qps[key] = 1
                cs.exit_section()
                
                if self.pQps.peer_wating_tasks[key]:
                    cur_flow_id = self.pQps.peer_wating_tasks[key].pop(0)
                    self.pQps.peer_qps[key] = 0
                    self.insert_packets(channel_id, cur_flow_id)
                
                self.iteratable(channel_id)
            else:
                self.phy_iteratable(channel_id)   

    def init_recv_ready(self):
        recv_ready_flows = {}
        for flow in self._flow_models.values():
            if flow.src != self.id:
                continue
            if flow.chunk_id != 0:
                continue
            if not flow.parent_flow_id:
                continue
            cur = (flow.channel_id, tuple(flow.prev))
            if cur not in recv_ready_flows:
                recv_ready_flows[cur] = [flow.flow_id]
            else:
                flow_ids = recv_ready_flows[cur]
                flag = True
                for flow_id in flow_ids:
                    old_flow = self._flow_models[(flow.channel_id, flow_id)]
                    if old_flow.parent_flow_id == flow.parent_flow_id:
                        flag = False
                        break
                if flag:
                    recv_ready_flows[cur].append(flow.flow_id)

        for cur, flow_ids in recv_ready_flows.items():
            for flow_id in flow_ids:
                self.recv_ready(cur[0], flow_id)

        return True

    def recv_ready(self, channel_id: int, flow_id: int) -> bool:
        # 获取流模型
        logger = MockNcclLog.get_instance()

        flow_key = (channel_id, flow_id)
        if flow_key not in self._flow_models:
            return False
        flow_model = self._flow_models[flow_key]
        recv_prevs = flow_model.prev

        logger.write_log(
            NcclLogLevel.DEBUG,
            "recv_ready channel_id=%d flow_id=%d prev_count=%d",
            channel_id, flow_id, len(recv_prevs)
        )

        for recv_prev in recv_prevs:
            # 创建模拟请求（sim_request）
            rcv_req = sim_request(
                vnet=self.stream.current_queue_id,
                layer_num=self.layer_num
            )

            # 创建事件处理数据
            ehd = RecvPacketEventHandlerData(
                stream=self.stream,
                src_rank=self.stream.owner.id,
                event_type=EventType.PacketReceived,
                sender_node=recv_prev,
                data_size=1
            )
            ehd.flow_tag.child_flow_id = -1
            ehd.flow_tag.current_flow_id = -1
            ehd.flow_tag.channel_id = channel_id
            ehd.flow_tag.tag_id = (
                self.layer_num * flow_model.chunk_count * self.m_channels +
                flow_model.chunk_count * flow_model.channel_id
            )

            # 调用前端模拟接收函数
            self.stream.owner.front_end_sim_recv(
                src_rank=0,
                data=Sys.dummy_data,
                data_size=flow_model.flow_size,
                data_type=req_type_e.UINT8,
                dest_rank=recv_prev,
                channel_id=channel_id,
                request=rcv_req,
                handler=Sys.handle_event,
                event_data=ehd
            )

        return True

    def release_packets(self, channel_id: int, flow_id: int, message_size: int):
        """释放数据包到MA或NPU"""
        logger = MockNcclLog.get_instance()

        logger.write_log(
            NcclLogLevel.DEBUG,
            "id: %d finish release_packets",
            self.id
        )
        
        packet_bundle = PacketBundle(
            owner=self.stream.owner,
            stream=self.stream,
            data=[],  # 空列表对应C++的{}
            processed=self.processed,
            send_back=self.send_back,
            message_size=message_size,
            transmition=self.transmition,
            channel_id=channel_id,
            flow_id=flow_id
        )
        
        if self.NPU_to_MA:
            packet_bundle.send_to_MA()
        else:
            packet_bundle.send_to_NPU()

    def process_stream_count(self, channel_id: int):
        logger = MockNcclLog.get_instance()

        logger.write_log(
            NcclLogLevel.DEBUG,
            "NcclTreeFlowModel::process_stream_count channel_id %d _stream_count %d",
            channel_id, self._stream_count[channel_id]
        )
        
        if self.PHY_MTP:
            # PHY_MTP模式下直接减少发送包计数
            self.send_packets -= 1
        else:
            # 非PHY_MTP模式，使用临界区保护
            with self.FlowCriticalSection():  # 假设FlowCriticalSection是上下文管理器
                if self._stream_count[channel_id] > 0:
                    self._stream_count[channel_id] -= 1
                
                # 检查流状态并更新
                if self._stream_count[channel_id] == 0 and self.stream.state != StreamState.Dead:
                    self.stream.changeState(StreamState.Zombie)

    def reduce(self, channel_id: int, flow_id: int):
        """减少流计数并处理数据包队列"""
        self.process_stream_count(channel_id)
        
        if not self.PHY_MTP:
            # 非PHY_MTP模式下操作数据包队列
            key = (channel_id, flow_id)
            if key in self.packets and len(self.packets[key]) > 0:
                self.packets[key].pop(0) 


    def iteratable(self, channel_id: int) -> bool:
        """检查是否所有通道和数据包都已处理完成"""
        logger = MockNcclLog.get_instance()
        all_channel_finished = True
        all_packets_freed = True
        
        with self.FlowCriticalSection():  # 假设FlowCriticalSection是上下文管理器
            # 检查所有通道的流计数是否为0
            for i in range(self.m_channels):
                if i in self._stream_count and self._stream_count[i] != 0:
                    all_channel_finished = False
            
            # 检查所有free_packets是否为0
            for value in self.free_packets.values():
                if value != 0:
                    all_packets_freed = False
                    break
        
        # 如果所有通道和数据包都已处理完成，则退出
        if all_channel_finished and all_packets_freed:
            self.exit()
            return False
        
        return True

    def insert_packets(self, channel_id: int, flow_id: int):
        """插入数据包到队列并发送"""
        logger = MockNcclLog.get_instance()
        
        # 检查参数有效性
        assert channel_id < self.m_channels, f"Invalid channel_id: {channel_id}"
        if not self.enabled:
            return
        
        flow_key = (channel_id, flow_id)
        assert flow_key in self._flow_models, f"Flow not found: {flow_key}"
        
        f = self._flow_models[flow_key]
        
        # 检查通道的数据包计数器
        assert channel_id in self.zero_latency_packets, f"Channel not found in zero_latency_packets: {channel_id}"
        assert channel_id in self.non_zero_latency_packets, f"Channel not found in non_zero_latency_packets: {channel_id}"
        
        # 初始化计数器（如果需要）
        if self.zero_latency_packets[channel_id] == 0 and self.non_zero_latency_packets[channel_id] == 0:
            self.zero_latency_packets[channel_id] = self.parallel_reduce * 1
            self.non_zero_latency_packets[channel_id] = self.get_non_zero_latency_packets()
            self.toggle = not self.toggle
        
        current_receiver = f.dest
        current_sender = f.prev
        
        # 处理零延迟数据包
        if self.zero_latency_packets[channel_id] > 0:
            logger.write_log(NcclLogLevel.DEBUG, f"id: {self.id} zero_latency_packets[channel_id] > 0")
            message_size = f.flow_size
            
            # 创建并配置数据包
            packet = MyPacket(
                vnet=self.stream.current_queue_id,
                sender=current_sender[0],
                receiver=current_receiver,
                size=message_size,
                channel_id=channel_id,
                flow_id=flow_id
            )
            packet.set_flow_id(flow_id)
            packet.sender = None
            
            # 设置发送参数并释放数据包
            self.processed = False
            self.send_back = False
            self.NPU_to_MA = True
            self.release_packets(channel_id, flow_id, message_size)
            
            # 更新计数器
            self.zero_latency_packets[channel_id] -= 1
            logger.write_log(NcclLogLevel.DEBUG, f"id: {self.id} zero_latency_packets[channel_id]: {self.zero_latency_packets[channel_id]}")
            return
        
        # 处理非零延迟数据包
        elif self.non_zero_latency_packets[channel_id] > 0:
            logger.write_log(NcclLogLevel.DEBUG, f"id: {self.id} non_zero_latency_packets[channel_id] > 0")
            message_size = f.flow_size
            
            # 创建并配置数据包
            packet = MyPacket(
                vnet=self.stream.current_queue_id,
                sender=current_sender[0],
                receiver=current_receiver,
                size=message_size,
                channel_id=channel_id,
                flow_id=flow_id
            )
            packet.set_flow_id(flow_id)
            packet.sender = None
            
            # 根据通信类型设置处理标志
            if self.comType == ComType.Reduce_Scatter or (self.comType == ComType.All_Reduce and self.toggle):
                self.processed = True
            else:
                self.processed = False
            
            # 根据剩余数据包数量设置回送标志
            if self.non_zero_latency_packets[channel_id] <= self.parallel_reduce * 1:
                self.send_back = False
            else:
                self.send_back = True
            
            # 设置发送方向并释放数据包
            self.NPU_to_MA = False
            self.release_packets(channel_id, flow_id, message_size)
            
            # 更新计数器
            self.non_zero_latency_packets[channel_id] -= 1
            logger.write_log(NcclLogLevel.DEBUG, f"id: {self.id} non_zero_latency_packets[channel_id]: {self.non_zero_latency_packets[channel_id]}")
            return
        
        # 如果无法发送任何数据包，抛出异常
        raise RuntimeError("should not inject nothing!")

    def ready(self, channel_id: int, flow_id: int) -> bool:
        """处理流准备就绪事件"""
        logger = MockNcclLog.get_instance()
        
        # 检查流状态并更新
        if self.stream.state in (StreamState.Created, StreamState.Ready):
            self.stream.changeState(StreamState.Executing)
        
        # 检查是否满足发送条件
        flow_key = (channel_id, flow_id)
        if (not self.enabled or 
            flow_key not in self.packets or len(self.packets[flow_key]) == 0 or
            self._stream_count[channel_id] == 0):
            logger.write_log(NcclLogLevel.DEBUG, "NcclTreeFlowModel not ready!")
            return False
        
        # 获取队首数据包
        packet = self.packets[flow_key][0]
        
        # 获取接收端列表
        recv_prevs = self._flow_models[flow_key].prev
        
        # 处理每个接收端
        for recv_prev in recv_prevs:
            # 创建接收请求
            rcv_req = sim_request(
                vnet=self.stream.current_queue_id,
                layer_num=self.layer_num,
                req_count=packet.msg_size,
                tag=channel_id
            )
            
            # 创建接收事件数据
            ehd = RecvPacketEventHandlerData(
                stream=self.stream,
                src_rank=self.stream.owner.id,
                event_type=EventType.PacketReceived,
                sender_node=packet.preferred_vnet,
                data_size=packet.stream_num
            )
            
            # 设置流标签
            flow_model = self._flow_models[flow_key]
            if not flow_model.parent_flow_id or flow_model.conn_type == "RING":
                ehd.flow_tag.tag_id = (
                    self.layer_num * flow_model.chunk_count * self.m_channels +
                    flow_model.chunk_count * flow_model.channel_id +
                    flow_model.chunk_id
                )
            else:
                ehd.flow_tag.tag_id = (
                    self.layer_num * flow_model.chunk_count * self.m_channels +
                    flow_model.chunk_count * flow_model.channel_id +
                    flow_model.chunk_id + 1
                )
            
            ehd.flow_tag.channel_id = packet.channel_id
            
            # 如果有空闲数据包，则发送接收请求
            free_key = (channel_id, recv_prev)
            if free_key in self.free_packets and self.free_packets[free_key] > 0:
                self.stream.owner.front_end_sim_recv(
                    src_rank=0,
                    data=Sys.dummy_data,
                    data_size=rcv_req.req_count,
                    data_type=req_type_e.UINT8,
                    dest_rank=recv_prev,
                    tag=rcv_req.tag,
                    request=rcv_req,
                    handler=Sys.handle_event,
                    event_data=ehd
                )
        
        # 创建发送请求
        snd_req = sim_request(
            src_rank=self.id,
            dst_rank=packet.preferred_dest,
            tag=channel_id,
            req_type=req_type_e.UINT8,
            vnet=self.stream.current_queue_id,
            layer_num=self.layer_num,
            req_count=packet.msg_size
        )
        
        # 设置发送请求的流标签
        flow_model = self._flow_models[flow_key]
        snd_req.flow_tag.tag_id = (
            self.layer_num * flow_model.chunk_count * self.m_channels +
            flow_model.channel_id * flow_model.chunk_count +
            flow_model.chunk_id
        )
        snd_req.flow_tag.channel_id = channel_id
        snd_req.flow_tag.flow_size = flow_model.flow_size
        snd_req.flow_tag.current_flow_id = flow_id
        snd_req.flow_tag.chunk_id = flow_model.chunk_id
        snd_req.flow_tag.child_flow_id = -1
        snd_req.flow_tag.tree_flow_list = flow_model.child_flow_id
        snd_req.flow_tag.sender_node = self.id
        snd_req.flow_tag.receiver_node = packet.preferred_dest
        snd_req.flow_tag.pQps = self.pQps
        snd_req.flow_tag.nvls_on = (self.comType == ComType.All_Reduce_NVLS)
        
        # 创建发送事件数据
        send_ehd = SendPacketEventHandlerData(
            stream=self.stream,
            src_rank=self.id,
            dst_rank=packet.preferred_dest,
            channel_id=channel_id,
            event_type=EventType.PacketSentFinshed
        )
        
        # 发送数据包
        self.stream.owner.front_end_sim_send(
            src_rank=0,
            data=Sys.dummy_data,
            data_size=snd_req.req_count,
            data_type=req_type_e.UINT8,
            dest_rank=packet.preferred_dest,
            tag=snd_req.flow_tag.tag_id,
            request=snd_req,
            handler=Sys.handle_event,
            event_data=send_ehd
        )
        
        return True

    def exit(self):
        """处理流退出事件"""
        logger = MockNcclLog.get_instance()
        
        if self.PHY_MTP:
            # 记录结束时间和通信延迟
            now_us = int(time.time() * 1e6)
            logger.write_log(NcclLogLevel.DEBUG, f"NcclTreeFlowModel exit time {now_us}")
            
            self.end_time = time.time()
            duration = int((self.end_time - self.start_time) * 1e6)  # 微秒
            logger.write_log(NcclLogLevel.DEBUG, f"Communication Latency：{duration} us")
            
            # 同步并等待
            import mpi4py.MPI as MPI
            MPI.COMM_WORLD.Barrier()
            time.sleep(1)
        else:
            # 清空所有数据包队列
            for key in list(self.packets.keys()):
                if self.packets[key]:
                    self.packets[key].clear()
        
        # 进入下一个虚拟网络
        self.stream.owner.proceed_to_next_vnet_baseline(self.stream)
        logger.write_log(NcclLogLevel.DEBUG, "NcclTreeFlowModel exit")

    def phy_iteratable(self, channel_id: int) -> bool:
        """检查是否所有发送和接收都已完成"""
        all_send_finished = self.send_packets == 0
        all_recv_finished = self.recv_packets == 0
        exit_flag = all_send_finished and all_recv_finished

        if exit_flag:
            self.judge_exit_flag = True
            return False
        else:
            return True

    def phy_ready(self, channel_id: int, flow_id: int) -> bool:
        """处理物理层准备就绪事件"""
        if self.stream.state in (StreamState.Created, StreamState.Ready):
            self.stream.changeState(StreamState.Executing)

        flow = self._flow_models[(channel_id, flow_id)]
        recv_prevs = flow.prev

        for recv_prev in recv_prevs:
            rcv_req = sim_request(
                vnet=self.stream.current_queue_id,
                layer_num=self.layer_num,
                req_count=flow.flow_size,
                tag=channel_id
            )

            ehd = RecvPacketEventHandlerData(
                stream=self.stream,
                src_rank=self.stream.owner.id,
                event_type=EventType.PacketReceived,
                sender_node=self.stream.current_queue_id,
                data_size=1
            )

            ehd.flow_tag.child_flow_id = -1
            ehd.flow_tag.current_flow_id = -1

            flow_model = self._flow_models[(channel_id, flow_id)]
            if not flow_model.parent_flow_id or flow_model.conn_type == "RING":
                ehd.flow_tag.tag_id = (
                    self.layer_num * flow_model.chunk_count * self.m_channels +
                    flow_model.chunk_count * flow_model.channel_id +
                    flow_model.chunk_id
                )
            else:
                ehd.flow_tag.tag_id = (
                    self.layer_num * flow_model.chunk_count * self.m_channels +
                    flow_model.chunk_count * flow_model.channel_id +
                    flow_model.chunk_id + 1
                )

            ehd.flow_tag.channel_id = flow.channel_id

            if (channel_id, recv_prev) in self.free_packets and self.free_packets[(channel_id, recv_prev)] > 0:
                self.stream.owner.front_end_sim_recv(
                    src_rank=0,
                    data=Sys.dummy_data,
                    data_size=rcv_req.req_count,
                    data_type=req_type_e.UINT8,
                    dest_rank=recv_prev,
                    tag=rcv_req.tag,
                    request=rcv_req,
                    handler=Sys.handle_event,
                    event_data=ehd
                )

        snd_req = sim_request(
            src_rank=self.id,
            dst_rank=flow.dest,
            tag=channel_id,
            req_type=req_type_e.UINT8,
            vnet=self.stream.current_queue_id,
            layer_num=self.layer_num,
            req_count=flow.flow_size
        )

        flow_model = self._flow_models[(channel_id, flow_id)]
        snd_req.flow_tag.tag_id = (
            self.layer_num * flow_model.chunk_count * self.m_channels +
            flow_model.channel_id * flow_model.chunk_count +
            flow_model.chunk_id
        )
        snd_req.flow_tag.channel_id = channel_id
        snd_req.flow_tag.flow_size = flow_model.flow_size
        snd_req.flow_tag.current_flow_id = flow_id
        snd_req.flow_tag.chunk_id = flow_model.chunk_id
        snd_req.flow_tag.child_flow_id = -1
        snd_req.flow_tag.tree_flow_list = flow_model.child_flow_id
        snd_req.flow_tag.sender_node = self.id
        snd_req.flow_tag.receiver_node = flow.dest
        snd_req.flow_tag.pQps = self.pQps
        snd_req.flow_tag.nvls_on = (self.comType == ComType.All_Reduce_NVLS)

        send_ehd = SendPacketEventHandlerData(
            stream=self.stream,
            src_rank=self.id,
            dst_rank=flow.dest,
            channel_id=channel_id,
            event_type=EventType.PacketSentFinshed
        )

        self.stream.owner.front_end_sim_send(
            src_rank=0,
            data=Sys.dummy_data,
            data_size=snd_req.req_count,
            data_type=req_type_e.UINT8,
            dest_rank=flow.dest,
            tag=snd_req.tag,
            request=snd_req,
            handler=Sys.handle_event,
            event_data=send_ehd
        )

        return True

    def waiting_to_exit(self):
        logger = MockNcclLog.get_instance()
        logger.write_log(NcclLogLevel.DEBUG, "NcclTreeFlowModel::waiting_to_exit begin ")
        
        while not self.judge_exit_flag:
            time.sleep(0.1)  # 避免空循环占用过多CPU资源

        self.exit()
        return