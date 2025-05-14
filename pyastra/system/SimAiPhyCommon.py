from dataclasses import dataclass
from typing import Any

TEST_IO_DEPTH = 16
MAX_CHILD_FLOW_SIZE = 20
NCCL_QPS_PER_PEER = 1
INIT_RECV_WR_NUMS = 1024
SEND_CHUNK_SIZE = 1024 * 1024
WR_NUMS = 1

def assert_non_null(x):
    assert x is not None, "Assertion failed: expected non-null value"

@dataclass
class MrInfo:
    addr: int
    len: int
    lkey: int
    rkey: int

class TransportData:
    channel_id: int
    chunk_id: int
    current_flow_id: int
    child_flow_id: int
    sender_node: int
    receiver_node: int
    flow_size: int
    pQps: Any  
    tag_id: int
    nvls_on: bool
    child_flow_list: list[int] = None 

    def __init__(
        self,
        channel_id: int,
        chunk_id: int,
        current_flow_id: int,
        child_flow_id: int,
        sender_node: int,
        receiver_node: int,
        flow_size: int,  
        pQps: object,
        tag_id: int,
        nvls_on: bool
    ):
        self.channel_id = channel_id
        self.chunk_id = chunk_id
        self.current_flow_id = current_flow_id
        self.child_flow_id = child_flow_id
        self.sender_node = sender_node
        self.receiver_node = receiver_node
        self.flow_size = flow_size
        self.pQps = pQps
        self.tag_id = tag_id
        self.nvls_on = nvls_on

if __name__ == "__main__":
    assert_non_null("valid object")
    
    mr = MrInfo(addr=0x1000, len=4096, lkey=123, rkey=456)
    
    td = TransportData(
        channel_id=0,
        chunk_id=1,
        current_flow_id=1001,
        child_flow_id=101,
        sender_node=2,
        receiver_node=3,
        flow_size=1024*1024,
        pQps=None,
        tag_id=42,
        nvls_on=True,
        child_flow_list=[101, 102, 103] 
    )
    print(td)