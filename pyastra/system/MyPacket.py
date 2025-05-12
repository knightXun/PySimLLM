# ******************************************************************************
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ******************************************************************************

from typing import Optional
from .Callable import Callable  # 假设存在Callable基类
from .Common import EventType, Tick  # 假设存在相关类型定义

class MyPacket(Callable):
    def __init__(self):
        # 基础初始化（对应C++无参构造）
        self.cycles_needed: int = 0
        self.fm_id: Optional[int] = None
        self.stream_num: Optional[int] = None
        self.notifier: Optional[Callable] = None
        self.sender: Optional[Callable] = None
        self.preferred_vnet: Optional[int] = None
        self.preferred_dest: Optional[int] = None
        self.preferred_src: Optional[int] = None
        self.msg_size: uint64 = 0  # Python中用int代替uint64
        self.ready_time: Optional[Tick] = None
        self.flow_id: Optional[int] = None
        self.parent_flow_id: Optional[int] = None
        self.child_flow_id: Optional[int] = None
        self.channel_id: Optional[int] = None
        self.chunk_id: Optional[int] = None

    @classmethod
    def from_vnet_src_dest(cls, preferred_vnet: int, preferred_src: int, preferred_dest: int) -> 'MyPacket':
        """对应C++构造函数：MyPacket(int preferred_vnet, int preferred_src, int preferred_dest)"""
        obj = cls()
        obj.preferred_vnet = preferred_vnet
        obj.preferred_src = preferred_src
        obj.preferred_dest = preferred_dest
        obj.msg_size = 0  # 初始消息大小为0
        return obj

    @classmethod
    def from_msg_vnet_src_dest(
        cls, 
        msg_size: int, 
        preferred_vnet: int, 
        preferred_src: int, 
        preferred_dest: int
    ) -> 'MyPacket':
        """对应C++构造函数：MyPacket(uint64_t msg_size, int preferred_vnet, int preferred_src, int preferred_dest)"""
        obj = cls()
        obj.msg_size = msg_size
        obj.preferred_vnet = preferred_vnet
        obj.preferred_src = preferred_src
        obj.preferred_dest = preferred_dest
        return obj

    @classmethod
    def from_full_params(
        cls, 
        preferred_vnet: int, 
        preferred_src: int, 
        preferred_dest: int, 
        msg_size: int, 
        channel_id: int, 
        flow_id: int
    ) -> 'MyPacket':
        """对应C++构造函数：MyPacket(int preferred_vnet, int preferred_src, int preferred_dest, uint64_t msg_size, int channel_id, int flow_id)"""
        obj = cls()
        obj.preferred_vnet = preferred_vnet
        obj.preferred_src = preferred_src
        obj.preferred_dest = preferred_dest
        obj.msg_size = msg_size
        obj.channel_id = channel_id
        obj.flow_id = flow_id
        return obj

    def set_flow_id(self, flow_id: int) -> None:
        """设置flow_id"""
        self.flow_id = flow_id

    def set_notifier(self, c: Callable) -> None:
        """设置通知回调对象"""
        self.notifier = c

    def call(self, event: EventType, data) -> None:
        """事件回调方法"""
        self.cycles_needed = 0
        if self.notifier is not None:
            # 调用通知对象的call方法，传递General事件类型和None数据
            self.notifier.call(EventType.General, None)
