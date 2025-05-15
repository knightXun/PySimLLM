import os
import socket
import struct
import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from mpi4py import MPI
import logging
import threading
import fcntl

from mpi4py import MPIc

import pyverbs
from pyverbs.context import Context
import pyverbs.device as d
from pyverbs.qp import QP, QPCap, QPInitAttr, QPAttr
from pyverbs.pd import PD
from pyverbs.cq import CQ
from pyverbs.mr import MR
from pyverbs.addr import GID
from pyverbs.wr import SendWR, RecvWR, SGE
from pyverbs.gid import GID
from pyverbs.gid import GID
from pyverbs.enums import (IBV_QPT_RC, IBV_ACCESS_LOCAL_WRITE, 
                          IBV_ACCESS_REMOTE_READ, IBV_ACCESS_REMOTE_WRITE,
                          IBV_MTU_256, IBV_SEND_SIGNALED, IBV_WR_RDMA_WRITE,
                          IBV_WR_RDMA_WRITE_WITH_IMM)

from SimAiPhyCommon import * 
from AstraNetworkAPI import *
from SimAiFlowModelRdma import *
from PhyMultiThread import * 
from MockNcclLog import * 
from BootStrapnet import *


@dataclass
class ibv_hand_shake:
    gid_index: int = 0
    qp_num: int = 0
    psn: int = 0
    lid: int = 0
    my_gid: bytes = bytes(16)  # 16字节GID
    recv_mr: MrInfo = MrInfo()
    send_mr: MrInfo = MrInfo()

class FlowPhyRdma:
    def __init__(self):
        pass 
    
    def __init__(self, gid_index: int):
        self.gid_index = gid_index
        self.g_ibv_ctx: Optional[pyverbs.Context] = None  
        self.ibv_peer_qps: Dict[Tuple[int, int], Dict[int, List[IbvQpContext]]] = {}  
        self.qpn2ctx: Dict[int, IbvQpContext] = {}  
        self.ibv_send_wr_map: Dict[Tuple[int, int], SendWR] = {} 
        self.ibv_recv_wr_id_map: Dict[int, int] = {}  
        self.ibv_send_wr_id_map: Dict[int, int] = {}
        self.logger = MockNcclLog()

    def __del__(self):
        self.ibv_fini()

    def send_wr_id_to_buff(self, qpn, wr_id):
        nccl_log = MockNcclLog.get_instance()
        
        # 获取工作请求
        key = (qpn, wr_id)
        send_wr = self.ibv_send_wr_map.get(key)
        if not send_wr:
            raise ValueError(f"No send_wr found for QPN={qpn}, WR_ID={wr_id}")

        # 获取缓冲区地址
        buff = send_wr.sg_list[0].addr
        ptrsendata = buff  # 直接转换为TransportData对象（假设已正确初始化）

        # 构造流标签
        flow_tag = ncclFlowTag(
            ptrsendata.channel_id,
            ptrsendata.chunk_id,
            ptrsendata.current_flow_id,
            ptrsendata.child_flow_id,
            ptrsendata.sender_node,
            ptrsendata.receiver_node,
            ptrsendata.flow_size,
            ptrsendata.pQps,
            ptrsendata.tag_id,
            ptrsendata.nvls_on
        )

        # 记录日志
        log_msg = (
            "SimAiFlowModelRdma.cc::send_wr_id_to_buff 数据包 send cqe,"
            "src_id %d dst_id %d qpn %d wr_id %d remote_addr %d len %d "
            "flow_id %d channel_id %d message_count: %d" % (
                flow_tag.sender_node,
                flow_tag.receiver_node,
                qpn,
                wr_id,
                send_wr.wr.rdma.remote_addr,
                send_wr.sg_list[0].length,
                flow_tag.current_flow_id,
                flow_tag.channel_id,
                flow_tag.flow_size
            )
        )
        nccl_log.write_log(NcclLogLevel.DEBUG, log_msg)

        return buff

    def recv_wr_id_to_buff(self, qpn, wr_id, chunk_id):
        nccl_log = MockNcclLog.get_instance()
        
        self.insert_recv_wr(qpn)
        
        qp = self.qpn2ctx.get(qpn)
        if not qp:
            raise ValueError(f"No QP context found for QPN={qpn}")

        base_addr = qp.src_info.recv_mr.addr
        recv_addr = base_addr + chunk_id * qp.chunk_size
        
        buff = self._addr_to_obj(recv_addr)  
        ptrsendata = buff
        
        flow_tag = ncclFlowTag(
            ptrsendata.channel_id,
            ptrsendata.chunk_id,
            ptrsendata.current_flow_id,
            ptrsendata.child_flow_id,
            ptrsendata.sender_node,
            ptrsendata.receiver_node,
            ptrsendata.flow_size,
            ptrsendata.pQps,
            ptrsendata.tag_id,
            ptrsendata.nvls_on
        )

        log_msg = (
            "SimAiFlowModelRdma.py::recv_wr_id_to_buff 数据包 recv cqe,"
            "src_id %d dst_id %d qpn %d wr_id %d local_addr %d "
            "flow_id %d channel_id %d message_count: %d" % (
                flow_tag.sender_node,
                flow_tag.receiver_node,
                qpn,
                wr_id,
                recv_addr,          # 注意：Python没有long long，可能需要调整格式
                flow_tag.current_flow_id,
                flow_tag.channel_id,
                flow_tag.flow_size
            )
        )
        nccl_log.write_log(NcclLogLevel.DEBUG, log_msg)

        return buff
    
    @staticmethod
    def modify_qo_to_rts(qp_ctx):
        NcclLog = MockNcclLog.get_instance()
        rc = 0
        
        attr = IbvQpAttr()
        attr.qp_state = IBV_QPS_INIT
        attr.port_num = IB_PORT
        attr.pkey_index = 0
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE
        flags_init = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
        rc = ibv_modify_qp(qp_ctx.qp, attr, flags_init)
        if rc != 0:
            NcclLog.writeLog(NcclLogLevel.ERROR, "failed to modify QP state to INIT")
            return rc

        # modify the QP to RTR 
        attr = IbvQpAttr()
        attr.qp_state = IBV_QPS_RTR
        attr.path_mtu = IBV_MTU_256
        attr.dest_qp_num = qp_ctx.dest_info.qp_num
        attr.rq_psn = qp_ctx.dest_info.psn
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 0x12
        attr.ah_attr.is_global = 1
        attr.ah_attr.dlid = qp_ctx.dest_info.lid
        attr.ah_attr.port_num = IB_PORT
        attr.ah_attr.grh.dgid = qp_ctx.dest_info.my_gid
        attr.ah_attr.grh.sgid_index = qp_ctx.src_info.gid_index
        attr.ah_attr.grh.flow_label = 0
        attr.ah_attr.grh.hop_limit = 1
        attr.ah_attr.grh.sgid_index = qp_ctx.src_info.gid_index
        attr.ah_attr.grh.traffic_class = 0
        
        remote_gid = attr.ah_attr.grh.dgid
        log_msg = (
            "remote_lid 0x%x local_gidindex %d remote_qpn %d remote_psn %d remote Gid " +
            ":".join(["%02x"]*16)
        ) % (
            attr.ah_attr.dlid, attr.ah_attr.grh.sgid_index,
            attr.dest_qp_num, attr.rq_psn, *tuple(remote_gid)
        )
        NcclLog.writeLog(NcclLogLevel.DEBUG, log_msg)
        
        flags_rtr = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | \
                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER

        rc = ibv_modify_qp(qp_ctx.qp, attr, flags_rtr)
        if rc != 0:
            NcclLog.writeLog(NcclLogLevel.ERROR, "failed to modify QP state to RTR")
            return rc
        else:
            NcclLog.writeLog(NcclLogLevel.DEBUG, "success to modify QP state to RTR")
        
        # Step 3: modify the QP to RTS
        attr = IbvQpAttr()
        attr.qp_state = IBV_QPS_RTS
        attr.timeout = 0x12
        attr.retry_cnt = 6
        attr.rnr_retry = 0
        attr.sq_psn = 0
        attr.max_rd_atomic = 1
        flags_rts = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | \
                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC
        rc = ibv_modify_qp(qp_ctx.qp, attr, flags_rts)
        if rc != 0:
            NcclLog.writeLog(NcclLogLevel.ERROR, "failed to modify QP state to RTS")
        else:
            NcclLog.writeLog(NcclLogLevel.DEBUG, "success to modify QP state to RTS")
        
        return rc

    def ibv_qp_conn(rank: int, src_rank: int, dst_rank: int, tag_id: int, send_data: pyverbs.IbvHandShake) -> pyverbs.IbvHandShake:
        """通过MPI交换ibv_hand_shake结构体的Python实现"""
        # 创建mr_info类型的MPI自定义数据类型（对应C的struct mr_info）
        # 1. 计算各字段的偏移量（使用ctypes模拟C结构体布局）
        class CMrInfo(ctypes.Structure):
            _fields_ = [
                ("addr", ctypes.c_uint64),
                ("len", ctypes.c_uint64),
                ("lkey", ctypes.c_uint32),
                ("rkey", ctypes.c_uint32)
            ]
        # 2. 创建MPI结构体类型
        mr_info_type = MPI.Datatype.Create_struct(
            blocklengths=[1, 1, 1, 1],
            displacements=[
                ctypes.offsetof(CMrInfo, "addr"),
                ctypes.offsetof(CMrInfo, "len"),
                ctypes.offsetof(CMrInfo, "lkey"),
                ctypes.offsetof(CMrInfo, "rkey")
            ],
            types=[MPI.UINT64_T, MPI.UINT64_T, MPI.UINT32_T, MPI.UINT32_T]
        )
        mr_info_type.Commit()

        # 创建my_gid类型的MPI自定义数据类型（16字节的uint8数组）
        mr_gid_type = MPI.UINT8_T.Create_contiguous(16)
        mr_gid_type.Commit()


        # 2. 创建MPI结构体类型
        hand_shake_type = MPI.Datatype.Create_struct(
            blocklengths=[1, 1, 1, 1, 1, 1, 1],
            displacements=[
                ctypes.offsetof(pyverbs.IbvHandShake, "gid_index"),
                ctypes.offsetof(pyverbs.IbvHandShake, "qp_num"),
                ctypes.offsetof(pyverbs.IbvHandShake, "psn"),
                ctypes.offsetof(pyverbs.IbvHandShake, "lid"),
                ctypes.offsetof(pyverbs.IbvHandShake, "my_gid"),
                ctypes.offsetof(pyverbs.IbvHandShake, "recv_mr"),
                ctypes.offsetof(pyverbs.IbvHandShake, "send_mr")
            ],
            types=[
                MPI.UINT32_T,    # gid_index (unsigned)
                MPI.UINT32_T,    # qp_num (unsigned)
                MPI.UINT32_T,    # psn (unsigned)
                MPI.UINT16_T,    # lid (uint16_t)
                mr_gid_type,     # my_gid (uint8_t[16])
                mr_info_type,    # recv_mr (struct mr_info)
                mr_info_type     # send_mr (struct mr_info)
            ]
        )
        hand_shake_type.Commit()

        # 执行MPI通信
        recv_data = pyverbs.IbvHandShake(
            gid_index=0, qp_num=0, psn=0, lid=0,
            my_gid=b"\x00"*16, recv_mr=MrInfo(0,0,0,0), send_mr=MrInfo(0,0,0,0)
        )

        if rank == src_rank:
            # 源进程：先发送后接收
            MPI.COMM_WORLD.Send([send_data, 1, hand_shake_type], dst_rank, tag_id)
            MPI.COMM_WORLD.Recv([recv_data, 1, hand_shake_type], dst_rank, tag_id)
        elif rank == dst_rank:
            # 目标进程：先接收后发送
            MPI.COMM_WORLD.Recv([recv_data, 1, hand_shake_type], src_rank, tag_id)
            MPI.COMM_WORLD.Send([send_data, 1, hand_shake_type], src_rank, tag_id)

        # 释放MPI自定义类型
        hand_shake_type.Free()
        mr_gid_type.Free()
        mr_info_type.Free()

        return recv_data
        

    def ibv_srv_alloc_ctx(
        rank: int,
        src_rank: int,
        dst_rank: int,
        channel_id: int,
        g_ibv_ctx: Context, 
        chunk_count: int,
        buffer_size: int,
        qp_nums: int
    ) -> list[IbvQpContext]:

        logger = MockNcclLog.get_instance()
        rc = 0
        qps = []
        
        # 1. 查询端口属性
        port_attr = g_ibv_ctx.query_port(1)  
        logger.write_log(
            "DEBUG", "src_rank %d dst_rank %d 获取端口属性成功", src_rank, dst_rank
        )

        # 2. 查询本地GID
        my_gid = Gid()
        try:
            g_ibv_ctx.query_gid(1, 0, my_gid)
        except Exception as e:
            logger.write_log(
                "DEBUG", "src_rank %d dst_rank %d 获取GID失败: %s", src_rank, dst_rank, str(e)
            )
            rc = 1

        # 3. 分配保护域（PD）
        try:
            pd = PD(g_ibv_ctx)
        except Exception as e:
            logger.write_log(
                "DEBUG", "src_rank %d dst_rank %d 分配PD失败: %s", src_rank, dst_rank, str(e)
            )
            rc = 1

        # 4. 创建完成队列（CQ）
        try:
            recv_cq = CQ(g_ibv_ctx, cq_sz=16384)
            send_cq = CQ(g_ibv_ctx, cq_sz=16384)
        except Exception as e:
            logger.write_log(
                "DEBUG", "src_rank %d dst_rank %d 创建CQ失败: %s", src_rank, dst_rank, str(e)
            )
            rc = 1

        # 5. 循环创建QP上下文
        for i in range(qp_nums):
            qp_ctx = IbvQpContext(
                chunk_size=buffer_size,
                recv_buf=None,
                send_buf=None,
                recv_mr=None,
                send_mr=None,
                qp=None,
                src_info=IbvHandShake(0, 0, 0, 0, b"", MrInfo(0,0,0,0), MrInfo(0,0,0,0)),
                dest_info=IbvHandShake(0, 0, 0, 0, b"", MrInfo(0,0,0,0), MrInfo(0,0,0,0))
            )

            # 5.1 分配接收/发送缓冲区（使用numpy数组模拟连续内存）
            try:
                qp_ctx.recv_buf = np.empty(chunk_count * buffer_size, dtype=np.uint8)
                qp_ctx.send_buf = np.empty(chunk_count * buffer_size, dtype=np.uint8)
            except MemoryError:
                logger.write_log(
                    "DEBUG", 
                    "src_rank %d dst_rank %d 分配缓冲区失败 chunk_count=%d buffer_size=%d",
                    src_rank, dst_rank, chunk_count, buffer_size
                )
                rc = 1

            # 5.2 注册内存区域（MR）
            mr_flags = Access.LOCAL_WRITE | Access.REMOTE_READ | Access.REMOTE_WRITE
            try:
                qp_ctx.recv_mr = Mr(pd, qp_ctx.recv_buf.nbytes, mr_flags, qp_ctx.recv_buf.ctypes.data)
                qp_ctx.send_mr = Mr(pd, qp_ctx.send_buf.nbytes, mr_flags, qp_ctx.send_buf.ctypes.data)
            except Exception as e:
                logger.write_log(
                    "DEBUG", "src_rank %d dst_rank %d 注册MR失败: %s", src_rank, dst_rank, str(e)
                )
                rc = 1

            qp_init_attr = QpInitAttr(
                qp_type=QPT_RC,
                send_cq=send_cq,
                recv_cq=recv_cq,
                cap=QpInitAttr.Cap(max_send_wr=512, max_recv_wr=16384, max_send_sge=1, max_recv_sge=1)
            )
            try:
                qp = QP(pd, qp_init_attr)
                qp_ctx.qp = qp
            except Exception as e:
                logger.write_log(
                    "DEBUG", "src_rank %d dst_rank %d 创建QP失败: %s", src_rank, dst_rank, str(e)
                )
                rc = 1

            # 5.4 填充本地QP信息（src_info）
            qp_ctx.src_info = IbvHandShake(
                gid_index=0,  # 根据实际gid_index调整
                qp_num=qp.qp_num,
                psn=0,
                lid=port_attr.lid,
                my_gid=my_gid.raw,  # GID的16字节原始数据
                recv_mr=MrInfo(
                    addr=qp_ctx.recv_buf.ctypes.data_as(ctypes.c_uint64).value,
                    len=qp_ctx.recv_buf.nbytes,
                    lkey=qp_ctx.recv_mr.lkey,
                    rkey=qp_ctx.recv_mr.rkey
                ),
                send_mr=MrInfo(
                    addr=qp_ctx.send_buf.ctypes.data_as(ctypes.c_uint64).value,
                    len=qp_ctx.send_buf.nbytes,
                    lkey=qp_ctx.send_mr.lkey,
                    rkey=qp_ctx.send_mr.rkey
                )
            )

            # 5.5 记录本地GID日志
            local_gid = qp_ctx.src_info.my_gid
            logger.write_log(
                "DEBUG",
                "src_rank %d dst_rank %d local_lid 0x%x local_gidindex %d local_qpn %d local_psn %d local Gid %s",
                src_rank,
                dst_rank,
                qp_ctx.src_info.lid,
                qp_ctx.src_info.gid_index,
                qp_ctx.src_info.qp_num,
                qp_ctx.src_info.psn,
                ":".join(f"{b:02x}" for b in local_gid)
            )

            # 5.6 执行QP连接握手（调用之前翻译的ibv_qp_conn）
            try:
                qp_ctx.dest_info = ibv_qp_conn(
                    rank, src_rank, dst_rank, channel_id, qp_ctx.src_info
                )
            except Exception as e:
                logger.write_log(
                    "DEBUG", "src_rank %d dst_rank %d QP握手失败: %s", src_rank, dst_rank, str(e)
                )
                rc = 1

            # 5.7 记录远程GID日志
            remote_gid = qp_ctx.dest_info.my_gid
            logger.write_log(
                "DEBUG",
                "src_rank %d dst_rank %d remote_lid 0x%x remote_gidindex %d remote_qpn %d remote_psn %d remote Gid %s",
                src_rank,
                dst_rank,
                qp_ctx.dest_info.lid,
                qp_ctx.dest_info.gid_index,
                qp_ctx.dest_info.qp_num,
                qp_ctx.dest_info.psn,
                ":".join(f"{b:02x}" for b in remote_gid)
            )

            # 5.8 修改QP状态到RTS（Ready To Send）
            try:
                modify_qo_to_rts(qp_ctx)  # 需实现状态转换逻辑（见下方说明）
            except Exception as e:
                logger.write_log(
                    "DEBUG", "src_rank %d dst_rank %d QP状态转换失败: %s", src_rank, dst_rank, str(e)
                )
                rc = 1

            # 5.9 错误处理（释放资源）
            if rc:
                if qp_ctx.qp:
                    qp_ctx.qp.destroy()
                if qp_ctx.recv_mr:
                    qp_ctx.recv_mr.dereg()
                if qp_ctx.send_mr:
                    qp_ctx.send_mr.dereg()
                # 注意：numpy数组内存自动管理，无需手动释放

            # 5.10 保存上下文
            qps.append(qp_ctx)
            qpn2ctx[qp_ctx.qp.qp_num] = qp_ctx 

        return qps

    def simai_ibv_post_send(
        self,
        channel_id: int,
        src_rank: int,
        dst_rank: int,
        send_buf: bytes,  
        length: int,      
        data_size: int,   
        chunk_id: int 
    ) -> bool:
        logger = MockNcclLog.get_instance()
        success = True

        # 1. 计算每个QP的缓冲区大小（假设NCCL_QPS_PER_PEER是类成员常量）
        buff_size_per_qp = data_size // self.NCCL_QPS_PER_PEER

        # 2. 解析传输数据头部（假设TransportData是已定义的数据类）
        ptrsendata = TransportData.from_buffer(send_buf)  # 需实现from_buffer方法解析字节流

        # 3. 遍历每个QP执行发送
        for qp_idx in range(self.NCCL_QPS_PER_PEER):
            # 3.1 获取QP上下文（假设ibv_peer_qps是类成员字典）
            qp_ctx: IbvQpContext = self.ibv_peer_qps.get((src_rank, dst_rank), {}).get(channel_id, [])[qp_idx]
            
            # 3.2 复制数据到QP发送缓冲区（内存视图操作）
            # 计算目标缓冲区位置：块ID * 每个QP的缓冲区大小
            dst_offset = chunk_id * buff_size_per_qp
            qp_ctx.send_buf[dst_offset : dst_offset + length] = send_buf[:length]

            # 3.3 生成唯一的WR ID（使用类成员字典维护）
            send_wr_id = self.ibv_send_wr_id_map.setdefault(qp_ctx.qp.qp_num, 0)
            self.ibv_send_wr_id_map[qp_ctx.qp.qp_num] += 1

            # 3.4 构造发送工作请求（WR）列表
            wr_list = []
            for wr_idx in range(self.WR_NUMS):
                # 3.4.1 构造SG列表（Scatter-Gather Entry）
                sge = Sge(
                    addr=qp_ctx.src_info.send_mr.addr + dst_offset + (wr_idx * (buff_size_per_qp // self.WR_NUMS)),
                    length=buff_size_per_qp // self.WR_NUMS,
                    lkey=qp_ctx.src_info.send_mr.lkey
                )

                # 3.4.2 构造发送WR
                if wr_idx != self.WR_NUMS - 1:
                    # 前N-1个WR：普通RDMA写
                    wr = SendWr(
                        wr_id=0,  # 仅最后一个WR需要唯一ID
                        sge=sge,
                        num_sge=1,
                        opcode=IBV_WR_RDMA_WRITE,
                        send_flags=0,
                        next_wr=None  # 后续手动链接
                    )
                else:
                    # 最后一个WR：带立即数的RDMA写（需要信号）
                    wr = SendWr(
                        wr_id=send_wr_id,
                        sge=sge,
                        num_sge=1,
                        opcode=IBV_WR_RDMA_WRITE_WITH_IMM,
                        send_flags=IBV_SEND_SIGNALED,
                        imm_data=chunk_id  # 立即数为块ID
                    )

                # 3.4.3 设置RDMA远程信息
                wr.rdma_remote_addr = qp_ctx.dest_info.recv_mr.addr + dst_offset
                wr.rdma_rkey = qp_ctx.dest_info.recv_mr.rkey

                wr_list.append(wr)

            # 3.5 链接WR链表（前N-1个WR的next指向后续WR）
            for wr_idx in range(self.WR_NUMS - 1):
                wr_list[wr_idx].next_wr = wr_list[wr_idx + 1]

            # 3.6 保存WR到映射（仅保存最后一个WR）
            self.ibv_send_wr_map[(qp_ctx.qp.qp_num, send_wr_id)] = wr_list[-1]

            # 3.7 提交发送请求
            try:
                qp_ctx.qp.post_send(wr_list[0])  # 传递链表头
            except Exception as e:
                logger.write_log(
                    "ERROR",
                    "post send failed, error: %s", str(e)
                )
                success = False
                continue

            # 3.8 记录发送日志
            now_us = int(time.time() * 1e6)  # 获取当前微秒时间戳
            logger.write_log(
                "DEBUG",
                "ibv_post_send qpn %d wr_id %d remote_addr 0x%x local_len %d channel_id %d flow_id %d time %lld",
                qp_ctx.qp.qp_num,
                send_wr_id,
                wr_list[-1].rdma_remote_addr,
                wr_list[-1].sge.length,
                ptrsendata.channel_id,
                ptrsendata.current_flow_id,
                now_us
            )

        return success

    def insert_recv_wr(self, qpn: int) -> bool:
        logger = MockNcclLog.get_instance()
        qp_ctx: IbvQpContext = self.qpn2ctx.get(qpn)
        if not qp_ctx:
            self.logger.write_log("ERROR", "QP上下文未找到，qpn=%d", qpn)
            return False

        # 生成唯一的接收WR ID（使用类成员字典维护）
        recv_wr_id = self.ibv_recv_wr_id_map.setdefault(qp_ctx.qp.qp_num, 0)
        self.ibv_recv_wr_id_map[qp_ctx.qp.qp_num] += 1

        # 构造接收工作请求（RecvWr）
        recv_wr = RecvWr(
            wr_id=recv_wr_id,
            sg_list=None,  # 对应C代码中的recv_wr.sg_list = nullptr
            num_sge=0      # 无SG列表（零拷贝接收场景）
        )

        # 记录日志
        logger.write_log(
            "DEBUG",
            "create_peer_qp, insert ibv_recv_wr_map elm, qpn %d recv_wr_id %d addr 0x%x len %d",
            qp_ctx.qp.qp_num,
            recv_wr_id,
            qp_ctx.src_info.recv_mr.addr,
            qp_ctx.src_info.recv_mr.len
        )

        # 提交接收请求到QP
        try:
            # 注意：pyverbs的post_recv接收RecvWr对象或列表
            # 若使用共享接收队列（SRQ），需传递srq参数：qp.post_recv(recv_wr, srq=srq)
            qp_ctx.qp.post_recv(recv_wr)
        except Exception as e:
            logger.write_log(
                "ERROR",
                "ibv_post_recv失败，qpn=%d, error: %s",
                qpn, str(e)
            )
            return False

        return True


    def init_recv_wr(self, qp: IbvQpContext, nums: int) -> bool:
        logger = MockNcclLog.get_instance()
        success = True
        
        for i in range(nums):
            recv_wr = RecvWr(
                wr_id=self.ibv_recv_wr_id_map[qp.qp.qp_num],  
                sg_list=None, 
                num_sge=0
            )
            
            # 记录WR ID并递增计数器
            current_wr_id = self.ibv_recv_wr_id_map[qp.qp.qp_num]
            self.ibv_recv_wr_id_map[qp.qp.qp_num] += 1
            
            logger.write_log(
                "DEBUG",
                "create_peer_qp, insert ibv_recv_wr_map elm, qpn %d recv_wr_id %d addr 0x%x len %d",
                qp.qp.qp_num,
                current_wr_id,
                qp.src_info.recv_mr.addr,
                qp.src_info.recv_mr.len
            )
            
            # 提交接收请求
            try:
                qp.qp.post_recv(recv_wr)
            except Exception as e:
                logger.write_log(
                    "ERROR",
                    "ibv_post_recv failed for qpn %d, wr_id %d: %s",
                    qp.qp.qp_num,
                    current_wr_id,
                    str(e)
                )
                success = False
        
        return success

    def ibv_create_peer_qp(
        self,
        rank: int,
        channel_id: int,
        src_rank: int,
        dst_rank: int,
        chunk_count: int,
        chunk_id: int,
        buffer_size: int
    ) -> bool:
        logger = MockNcclLog.get_instance()
        success = True
        
        # 计算每个QP的缓冲区大小
        buff_size_per_qp = buffer_size // self.NCCL_QPS_PER_PEER
        
        # 检查是否需要创建新的QP
        if (
            ((src_rank, dst_rank) not in self.ibv_peer_qps or 
            channel_id not in self.ibv_peer_qps[(src_rank, dst_rank)]) and
            ((dst_rank, src_rank) not in self.ibv_peer_qps or 
            channel_id not in self.ibv_peer_qps[(dst_rank, src_rank)])
        ):
            # 分配并初始化QP上下文
            qp_contexts = self.ibv_srv_alloc_ctx(
                rank,
                src_rank,
                dst_rank,
                channel_id,
                self.g_ibv_ctx,
                chunk_count,
                buff_size_per_qp,
                self.NCCL_QPS_PER_PEER
            )
            
            # 存储QP上下文（双向映射）
            self.ibv_peer_qps[(dst_rank, src_rank)][channel_id] = qp_contexts
            self.ibv_peer_qps[(src_rank, dst_rank)][channel_id] = qp_contexts
            
            # 启动发送和接收完成队列轮询线程
            send_cq = qp_contexts[0].qp.send_cq
            recv_cq = qp_contexts[0].qp.recv_cq
            
            send_thread = threading.Thread(
                target=self.create_polling_cqe_thread,
                args=(send_cq, 0)
            )
            send_thread.daemon = True  # 设置为守护线程
            send_thread.start()
            
            recv_thread = threading.Thread(
                target=self.create_polling_cqe_thread,
                args=(recv_cq, 0)
            )
            recv_thread.daemon = True  # 设置为守护线程
            recv_thread.start()
        
        # 记录创建信息
        logger.write_log(
            "DEBUG",
            "SimAiFlowModelRdma.cc create_peer_qp local_rank %d src %d dst %d channel_id %d",
            rank, src_rank, dst_rank, channel_id
        )
        
        # 如果当前节点是目标节点，初始化接收WR
        if dst_rank == rank:
            for i in range(self.NCCL_QPS_PER_PEER):
                qp = self.ibv_peer_qps[(src_rank, dst_rank)][channel_id][i]
                
                # 创建接收WR（使用之前实现的函数）
                success &= self.insert_recv_wr(qp.qp.qp_num)
        
        return success

    def zeta_util_ifname_to_inet_addr(ifname: str) -> str:

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            ifr = struct.pack('256s', bytes(ifname[:15], 'utf-8'))
            ifr = fcntl.ioctl(
                sock.fileno(),
                0x8915, 
                ifr
            )
            ip_addr = socket.inet_ntoa(ifr[20:24])
        return ip_addr

    def ibdev2netdev(indev: str) -> str:
        if indev.startswith("mlx5_bond_"):
            return "bond" + indev[10:]
        else:
            return indev

    def ibv_init(self) -> int:
        logger = MockNcclLog.get_instance()
        src_addr = self.rank2addr[self.local_rank]  # 假设rank2addr是类成员
        
        # 获取IB设备列表
        devices = Device.get_device_list()
        nb_dev = len(devices)
        if nb_dev == 0:
            logger.write_log("ERROR", "未找到IB设备")
            return -1

        ibv_dev = None
        for dev in devices:
            # 转换设备名称
            netdev = ibdev2netdev(dev.name)
            logger.write_log("DEBUG", f"netdev {netdev}, ibdev {dev.name}")
            
            # 获取接口IP
            ip = zeta_util_ifname_to_inet_addr(netdev)
            if ip == src_addr:
                ibv_dev = dev
                break

        if not ibv_dev:
            logger.write_log("ERROR", "未找到匹配的IB设备")
            return -1

        logger.write_log("DEBUG", f"IB设备 {ibv_dev.name} 初始化成功")
        
        # 打开设备上下文
        try:
            self.g_ibv_ctx = Context(ibv_dev)
        except Exception as e:
            logger.write_log("ERROR", f"打开IB设备失败: {str(e)}")
            return -1

        return 0


    def ibv_fini(self) -> int:
        logger = MockNcclLog.get_instance()
        
        for peer_pair, channel_dict in self.ibv_peer_qps.items():
            for channel_id, qp_contexts in channel_dict.items():
                for qp_ctx in qp_contexts:
                    if qp_ctx.qp.recv_cq:
                        qp_ctx.qp.recv_cq.close()  # pyverbs自动处理销毁
                        qp_ctx.qp.recv_cq = None
                    if qp_ctx.qp.send_cq:
                        qp_ctx.qp.send_cq.close()
                        qp_ctx.qp.send_cq = None
                    
                    # 释放保护域（PD）
                    if qp_ctx.qp.pd:
                        qp_ctx.qp.pd.close()
                        qp_ctx.qp.pd = None
                    
                    # 释放内存缓冲区（numpy数组自动管理内存）
                    qp_ctx.recv_buf = None
                    qp_ctx.send_buf = None
                    
                    # 销毁队列对（QP）
                    if qp_ctx.qp:
                        qp_ctx.qp.close()
                        qp_ctx.qp = None
        
        # 清空对等QP映射
        self.ibv_peer_qps.clear()
        return 0

    def _cleanup_qp_context(self, qp_ctx: IbvQpContext):
        """清理单个QP上下文资源"""
        if qp_ctx.qp:
            qp_ctx.qp.destroy()
        if qp_ctx.send_mr:
            qp_ctx.send_mr.dereg()
        if qp_ctx.recv_mr:
            qp_ctx.recv_mr.dereg()
        if qp_ctx.send_cq:
            qp_ctx.send_cq.destroy()
        if qp_ctx.recv_cq:
            qp_ctx.recv_cq.destroy()
        if qp_ctx.pd:
            qp_ctx.pd.dealloc()

    def _poll_cqe(self, cq: CQ):
        """完成队列轮询（简化示例）"""
        while True:
            try:
                cqes = cq.poll()
                for cqe in cqes:
                    self.logger.writeLog(logging.DEBUG, f"Received CQE: wr_id={cqes[0].wr_id}, status={cqes[0].status}")
            except Exception as e:
                self.logger.writeLog(logging.WARNING, f"CQ poll error: {str(e)}")


flow_rdma = FlowPhyRdma()

if __name__ == "__main__":
    # 示例用法（需在MPI环境中运行）
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 初始化RDMA实例
    rdma = FlowPhyRdma(gid_index=0)
    if rdma.ibv_init() != 0:
        exit(1)

    # 创建对端QP（示例：rank 0与rank 1通信）
    if rank == 0:
        rdma.ibv_create_peer_qp(
            rank=0,
            channel_id=0,
            src_rank=0,
            dst_rank=1,
            chunk_count=10,
            chunk_id=0,
            buffer_size=4096
        )
    elif rank == 1:
        rdma.ibv_create_peer_qp(
            rank=1,
            channel_id=0,
            src_rank=0,
            dst_rank=1,
            chunk_count=10,
            chunk_id=0,
            buffer_size=4096
        )

    # 发送数据（示例）
    if rank == 0:
        send_data = memoryview(bytearray(4096))  # 4KB数据
        rdma.simai_ibv_post_send(
            channel_id=0,
            src_rank=0,
            dst_rank=1,
            send_buf=send_data,
            data_size=4096,
            chunk_id=0
        )

    # 清理资源
    rdma.ibv_fini()