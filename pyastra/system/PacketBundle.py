import time
from typing import List, Optional
from astra_sim.system import Sys, BaseStream, MyPacket, MemBus, MockNcclLog, BasicEventHandlerData, EventType, Callable

class PacketBundle(Callable):
    def __init__(
        self,
        generator: Sys,
        stream: BaseStream,
        needs_processing: bool,
        send_back: bool,
        size: int,
        transmition: MemBus.Transmition,
        locked_packets: Optional[List[MyPacket]] = None,
        channel_id: int = -1,
        flow_id: Optional[int] = None
    ):
        self.generator = generator
        self.stream = stream
        self.needs_processing = needs_processing
        self.send_back = send_back
        self.size = size
        self.transmition = transmition
        self.locked_packets = locked_packets if locked_packets is not None else []
        self.channel_id = channel_id
        self.flow_id = flow_id
        self.creation_time = Sys.boosted_tick()
        self.delay: int = 0  # 初始化为0，具体值在call方法中计算

    def send_to_MA(self) -> None:
        self.generator.mem_bus.send_from_NPU_to_MA(
            self.transmition,
            self.size,
            self.needs_processing,
            self.send_back,
            self
        )

    def send_to_NPU(self) -> None:
        nccl_log = MockNcclLog.get_instance()
        self.generator.mem_bus.send_from_MA_to_NPU(
            self.transmition,
            self.size,
            self.needs_processing,
            self.send_back,
            self
        )
        nccl_log.write_log(MockNcclLog.NcclLogLevel.DEBUG, "send_to_NPU done")

    def call(self, event: EventType, data: Optional[BasicEventHandlerData]) -> None:
        nccl_log = MockNcclLog.get_instance()
        nccl_log.write_log(MockNcclLog.NcclLogLevel.DEBUG, "packet bundle call")

        if self.needs_processing:
            self.needs_processing = False
            # 计算延迟：mem_write(size) + mem_read(size) + mem_read(size)
            self.delay = (self.generator.mem_write(self.size) + 
                         self.generator.mem_read(self.size) + 
                         self.generator.mem_read(self.size))
            # 注册事件
            self.generator.try_register_event(
                self,
                EventType.CommProcessingFinished,
                data,
                self.delay
            )
            return

        current = Sys.boosted_tick()
        # 对应C++中的 #ifndef PHY_MTP 条件编译
        if not Sys.PHY_MTP:  # 假设Sys类有PHY_MTP静态属性控制该逻辑
            for packet in self.locked_packets:
                packet.ready_time = current

        # 创建事件处理数据（根据channel_id决定是否需要flow_id）
        if self.channel_id != -1 and self.flow_id is not None:
            ehd = BasicEventHandlerData(self.channel_id, self.flow_id)
        else:
            ehd = data  # 使用传入的data或保持None

        # 调用stream的call方法
        if self.channel_id == -1:
            self.stream.call(EventType.General, data)
        else:
            self.stream.call(EventType.NCCL_General, ehd)

        # Python中无需显式delete，由垃圾回收自动处理
