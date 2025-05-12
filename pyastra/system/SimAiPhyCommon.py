from dataclasses import dataclass
from typing import Any

# 常量定义（对应C++中的宏）
TEST_IO_DEPTH = 16
MAX_CHILD_FLOW_SIZE = 20
NCCL_QPS_PER_PEER = 1
INIT_RECV_WR_NUMS = 1024
SEND_CHUNK_SIZE = 1024 * 1024
WR_NUMS = 1

def assert_non_null(x):
    """对应C++中的assert_non_null宏"""
    assert x is not None, "Assertion failed: expected non-null value"

@dataclass
class MrInfo:
    """对应C++的mr_info结构体"""
    addr: int
    len: int
    lkey: int
    rkey: int

@dataclass
class TransportData:
    """对应C++的TransportData结构体"""
    channel_id: int
    chunk_id: int
    current_flow_id: int
    child_flow_id: int
    sender_node: int
    receiver_node: int
    flow_size: int
    pQps: Any  # 对应void*类型
    tag_id: int
    nvls_on: bool
    child_flow_list: list[int] = None  # 动态数组需要特殊处理

    def __post_init__(self):
        """初始化动态数组"""
        if self.child_flow_list is None:
            self.child_flow_list = [0] * MAX_CHILD_FLOW_SIZE
        # 确保数组长度不超过限制
        if len(self.child_flow_list) > MAX_CHILD_FLOW_SIZE:
            raise ValueError(f"child_flow_list exceeds maximum size {MAX_CHILD_FLOW_SIZE}")

# 示例用法
if __name__ == "__main__":
    # 测试assert_non_null
    assert_non_null("valid object")
    
    # 创建MrInfo实例
    mr = MrInfo(addr=0x1000, len=4096, lkey=123, rkey=456)
    
    # 创建TransportData实例
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
        child_flow_list=[101, 102, 103]  # 可选初始化
    )
    print(td)