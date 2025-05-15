
from typing import Optional
from .Callable import Callable  
from .Common import EventType, Tick  
# from Common import MyPacket

class MyPacket(Callable):
    def __init__(self):
        self.cycles_needed: int = 0
        self.fm_id: Optional[int] = None
        self.stream_num: Optional[int] = None
        self.notifier: Optional[Callable] = None
        self.sender: Optional[Callable] = None
        self.preferred_vnet: Optional[int] = None
        self.preferred_dest: Optional[int] = None
        self.preferred_src: Optional[int] = None
        self.msg_size = 0  
        self.ready_time: Optional[Tick] = None
        self.flow_id: Optional[int] = None
        self.parent_flow_id: Optional[int] = None
        self.child_flow_id: Optional[int] = None
        self.channel_id: Optional[int] = None
        self.chunk_id: Optional[int] = None

    @classmethod
    def from_vnet_src_dest(cls, preferred_vnet: int, preferred_src: int, preferred_dest: int):
        obj = cls()
        obj.preferred_vnet = preferred_vnet
        obj.preferred_src = preferred_src
        obj.preferred_dest = preferred_dest
        obj.msg_size = 0 
        return obj

    @classmethod
    def from_msg_vnet_src_dest(
        cls, 
        msg_size: int, 
        preferred_vnet: int, 
        preferred_src: int, 
        preferred_dest: int
    ):
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
    ):
        obj = cls()
        obj.preferred_vnet = preferred_vnet
        obj.preferred_src = preferred_src
        obj.preferred_dest = preferred_dest
        obj.msg_size = msg_size
        obj.channel_id = channel_id
        obj.flow_id = flow_id
        return obj

    def set_flow_id(self, flow_id: int) -> None:
        self.flow_id = flow_id

    def set_notifier(self, c: Callable) -> None:
        self.notifier = c

    def call(self, event: EventType, data: Callable) -> None:
        self.cycles_needed = 0
        if self.notifier is not None:

            self.notifier.call(EventType.General, None)
