import os
import logging
import socket
import struct
from dataclasses import dataclass
from mpi4py import MPI
from pyverbs.context import Context
from pyverbs.pd import PD
from pyverbs.cq import CQ, CompChannel
from pyverbs.qp import QP, QPInitAttr, QPAttr, QPS
from pyverbs.mr import MR
from pyverbs.addr import AH, AHAttr
from pyverbs.gid import GID
from pyverbs.enums import (IBV_QPT_RC, IBV_ACCESS_LOCAL_WRITE, 
                          IBV_ACCESS_REMOTE_READ, IBV_ACCESS_REMOTE_WRITE,
                          IBV_MTU_256, IBV_SEND_SIGNALED, IBV_WR_RDMA_WRITE,
                          IBV_WR_RDMA_WRITE_WITH_IMM)

# 日志配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class mr_info:
    addr: int = 0
    len: int = 0
    lkey: int = 0
    rkey: int = 0

@dataclass
class ibv_hand_shake:
    gid_index: int = 0
    qp_num: int = 0
    psn: int = 0
    lid: int = 0
    my_gid: bytes = bytes(16)  # 16字节GID
    recv_mr: mr_info = mr_info()
    send_mr: mr_info = mr_info()

class FlowPhyRdma:
    def __init__(self, gid_index: int):
        self.gid_index = gid_index
        self.g_ibv_ctx: Context = None
        self.ibv_peer_qps = {}  # 结构：{(src_rank, dst_rank): {channel_id: [qp_contexts...]}}
        self.qpn2ctx = {}       # {qp_num: qp_context}
        self.ibv_send_wr_id_map = {}  # {qp_num: next_wr_id}
        self.ibv_recv_wr_id_map = {}  # {qp_num: next_wr_id}
        self.mpi_comm = MPI.COMM_WORLD
        self.rank = self.mpi_comm.Get_rank()

    def __del__(self):
        self.ibv_fini()

    def send_wr_id_to_buff(self, qpn: int, wr_id: int) -> bytes:
        # 实现类似C++的日志记录和缓冲区映射逻辑
        send_wr = self.ibv_send_wr_map.get((qpn, wr_id))
        if not send_wr:
            logger.error(f"Send WR not found: qpn={qpn}, wr_id={wr_id}")
            return None
        return send_wr['sg_list'][0]['addr']

    def recv_wr_id_to_buff(self, qpn: int, wr_id: int, chunk_id: int) -> bytes:
        self.insert_recv_wr(qpn)
        qp_ctx = self.qpn2ctx.get(qpn)
        if not qp_ctx:
            logger.error(f"QP context not found: qpn={qpn}")
            return None
        recv_addr = qp_ctx['recv_mr'].addr + chunk_id * qp_ctx['chunk_size']
        return recv_addr.to_bytes(8, 'big')  # 模拟地址转换

    def modify_qp_to_rts(self, qp_ctx: dict) -> bool:
        """修改QP状态到RTS"""
        try:
            # 修改到INIT状态
            attr = QPAttr(qp_state=QPS.IBV_QPS_INIT, 
                        pkey_index=0, 
                        port_num=1, 
                        access_flags=IBV_ACCESS_LOCAL_WRITE | 
                                    IBV_ACCESS_REMOTE_READ | 
                                    IBV_ACCESS_REMOTE_WRITE)
            qp_ctx['qp'].modify(attr, 'state,pkey_index,port,access_flags')

            # 修改到RTR状态
            ah_attr = AHAttr(dlid=qp_ctx['dest_info'].lid, 
                            port_num=1, 
                            is_global=1, 
                            sgid_index=self.gid_index)
            ah = AH(qp_ctx['pd'], ah_attr)
            
            attr = QPAttr(qp_state=QPS.IBV_QPS_RTR,
                        path_mtu=IBV_MTU_256,
                        dest_qp_num=qp_ctx['dest_info'].qp_num,
                        rq_psn=qp_ctx['dest_info'].psn,
                        ah=ah)
            qp_ctx['qp'].modify(attr, 'state,path_mtu,dest_qp_num,rq_psn,ah')

            # 修改到RTS状态
            attr = QPAttr(qp_state=QPS.IBV_QPS_RTS,
                        timeout=0x12,
                        retry_cnt=6,
                        rnr_retry=0,
                        sq_psn=0)
            qp_ctx['qp'].modify(attr, 'state,timeout,retry_cnt,rnr_retry,sq_psn')
            return True
        except Exception as e:
            logger.error(f"QP state transition failed: {str(e)}")
            return False

    def ibv_qp_conn(self, src_rank: int, dst_rank: int, tag_id: int, 
                   send_data: ibv_hand_shake) -> ibv_hand_shake:
        """MPI握手交换RDMA信息"""
        if self.rank == src_rank:
            self.mpi_comm.send(send_data, dest=dst_rank, tag=tag_id)
            recv_data = self.mpi_comm.recv(source=dst_rank, tag=tag_id)
        elif self.rank == dst_rank:
            recv_data = self.mpi_comm.recv(source=src_rank, tag=tag_id)
            self.mpi_comm.send(send_data, dest=src_rank, tag=tag_id)
        else:
            return None
        return recv_data

    def ibv_srv_alloc_ctx(self, src_rank: int, dst_rank: int, channel_id: int,
                         chunk_count: int, buffer_size: int, qp_nums: int) -> list:
        """分配QP上下文"""
        qps = []
        try:
            # 获取IB设备上下文
            ctx = Context(name=os.environ.get('IB_DEVICE', 'mlx5_0'))
            pd = PD(ctx)
            comp_chan = CompChannel(ctx)
            send_cq = CQ(ctx, 16384, comp_chan=comp_chan)
            recv_cq = CQ(ctx, 16384, comp_chan=comp_chan)

            # 查询本地LID和GID
            port_attr = ctx.query_port(1)
            my_gid = ctx.query_gid(1, self.gid_index)

            for _ in range(qp_nums):
                # 分配内存并注册MR
                send_buf = bytearray(chunk_count * buffer_size)
                recv_buf = bytearray(chunk_count * buffer_size)
                send_mr = MR(pd, send_buf, IBV_ACCESS_LOCAL_WRITE | 
                            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE)
                recv_mr = MR(pd, recv_buf, IBV_ACCESS_LOCAL_WRITE | 
                            IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE)

                # 初始化QP属性
                qp_init_attr = QPInitAttr(qp_type=IBV_QPT_RC,
                                        send_cq=send_cq,
                                        recv_cq=recv_cq,
                                        cap=dict(max_send_wr=512, 
                                                max_recv_wr=16384,
                                                max_send_sge=1,
                                                max_recv_sge=1))
                qp = QP(pd, qp_init_attr)

                # 填充本地握手信息
                src_info = ibv_hand_shake(
                    gid_index=self.gid_index,
                    qp_num=qp.qp_num,
                    psn=0,
                    lid=port_attr.lid,
                    my_gid=my_gid.raw,
                    recv_mr=mr_info(addr=recv_mr.buf_addr, 
                                   len=recv_mr.length,
                                   lkey=recv_mr.lkey,
                                   rkey=recv_mr.rkey),
                    send_mr=mr_info(addr=send_mr.buf_addr,
                                   len=send_mr.length,
                                   lkey=send_mr.lkey,
                                   rkey=send_mr.rkey)
                )

                # 与对端握手
                dest_info = self.ibv_qp_conn(src_rank, dst_rank, channel_id, src_info)

                # 保存上下文
                qp_ctx = {
                    'qp': qp,
                    'chunk_size': buffer_size,
                    'send_buf': send_buf,
                    'recv_buf': recv_buf,
                    'send_mr': send_mr,
                    'recv_mr': recv_mr,
                    'src_info': src_info,
                    'dest_info': dest_info,
                    'pd': pd,
                    'send_cq': send_cq,
                    'recv_cq': recv_cq
                }

                # 状态转换
                if self.modify_qp_to_rts(qp_ctx):
                    qps.append(qp_ctx)
                    self.qpn2ctx[qp.qp_num] = qp_ctx
                    self.ibv_send_wr_id_map[qp.qp_num] = 0
                    self.ibv_recv_wr_id_map[qp.qp_num] = 0

            return qps

        except Exception as e:
            logger.error(f"QP allocation failed: {str(e)}")
            return []

    def simai_ibv_post_send(self, channel_id: int, src_rank: int, dst_rank: int,
                          send_buf: bytes, data_size: int, chunk_id: int) -> bool:
        """提交RDMA写操作"""
        try:
            qp_ctxs = self.ibv_peer_qps.get((src_rank, dst_rank), {}).get(channel_id, [])
            if not qp_ctxs:
                logger.error(f"No QPs found for {src_rank}->{dst_rank} channel {channel_id}")
                return False

            for qp_ctx in qp_ctxs:
                wr_id = self.ibv_send_wr_id_map[qp_ctx['qp'].qp_num]
                self.ibv_send_wr_id_map[qp_ctx['qp'].qp_num] += 1

                # 构造发送WR
                sge = [{'addr': qp_ctx['send_mr'].buf_addr + chunk_id * data_size,
                        'length': data_size,
                        'lkey': qp_ctx['send_mr'].lkey}]

                wr = {
                    'opcode': IBV_WR_RDMA_WRITE_WITH_IMM,
                    'send_flags': IBV_SEND_SIGNALED,
                    'imm_data': chunk_id,
                    'sg_list': sge,
                    'wr_id': wr_id,
                    'rdma': {
                        'remote_addr': qp_ctx['dest_info'].recv_mr.addr + chunk_id * data_size,
                        'rkey': qp_ctx['dest_info'].recv_mr.rkey
                    }
                }

                # 提交发送请求
                qp_ctx['qp'].post_send(wr)
                logger.debug(f"Posted send: qpn={qp_ctx['qp'].qp_num}, wr_id={wr_id}")

            return True
        except Exception as e:
            logger.error(f"Post send failed: {str(e)}")
            return False

    def init_recv_wr(self, qpn: int, nums: int) -> bool:
        """初始化接收工作请求"""
        qp_ctx = self.qpn2ctx.get(qpn)
        if not qp_ctx:
            logger.error(f"QP context not found: qpn={qpn}")
            return False

        for _ in range(nums):
            wr_id = self.ibv_recv_wr_id_map[qp_ctx['qp'].qp_num]
            self.ibv_recv_wr_id_map[qp_ctx['qp'].qp_num] += 1

            wr = {
                'wr_id': wr_id,
                'sg_list': [{'addr': qp_ctx['recv_mr'].buf_addr,
                            'length': qp_ctx['recv_mr'].length,
                            'lkey': qp_ctx['recv_mr'].lkey}]
            }

            qp_ctx['qp'].post_recv(wr)
            logger.debug(f"Posted recv: qpn={qpn}, wr_id={wr_id}")

        return True

    def ibv_init(self) -> int:
        """初始化IB设备"""
        try:
            # 自动检测IB设备（需要环境变量IB_DEVICE指定设备名）
            self.g_ibv_ctx = Context(name=os.environ.get('IB_DEVICE', 'mlx5_0'))
            logger.debug(f"IB device initialized: {self.g_ibv_ctx.name}")
            return 0
        except Exception as e:
            logger.error(f"IB initialization failed: {str(e)}")
            return -1

    def ibv_fini(self) -> int:
        """清理资源"""
        for (src, dst), channels in self.ibv_peer_qps.items():
            for channel_id, qp_ctxs in channels.items():
                for qp_ctx in qp_ctxs:
                    if qp_ctx['qp']:
                        qp_ctx['qp'].destroy()
                    if qp_ctx['send_mr']:
                        qp_ctx['send_mr'].dereg()
                    if qp_ctx['recv_mr']:
                        qp_ctx['recv_mr'].dereg()
                    if qp_ctx['send_cq']:
                        qp_ctx['send_cq'].destroy()
                    if qp_ctx['recv_cq']:
                        qp_ctx['recv_cq'].destroy()
                    if qp_ctx['pd']:
                        qp_ctx['pd'].dealloc()
        self.ibv_peer_qps.clear()
        self.qpn2ctx.clear()
        logger.debug("IB resources cleaned up")
        return 0

if __name__ == "__main__":
    # 使用示例
    flow_rdma = FlowPhyRdma(gid_index=0)
    if flow_rdma.ibv_init() != 0:
        exit(1)

    # 创建QP对（示例参数）
    flow_rdma.ibv_peer_qps[(0, 1)] = {
        0: flow_rdma.ibv_srv_alloc_ctx(0, 1, 0, chunk_count=4, buffer_size=4096, qp_nums=4)
    }

    # 发送数据示例
    test_data = b"test_rdma_data"
    flow_rdma.simai_ibv_post_send(0, 0, 1, test_data, len(test_data), 0)

    # 清理资源
    flow_rdma.ibv_fini()
    