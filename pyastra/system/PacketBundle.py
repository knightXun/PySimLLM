import time
from typing import List, Optional
from system import Sys, BaseStream, MyPacket, MemBus, MockNcclLog, BasicEventHandlerData, Callable
from Common import EventType

class PacketBundle(Callable):
    def __init__(
        self,
        generator: Sys,
        stream: BaseStream,
        needs_processing: bool,
        send_back: bool,
        size: int,
        transmition: MemBus.Transmition,
        locked_packets: Optional[List[MyPacket.MyPacket]] = None,
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
        self.delay: int = 0  

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

    def call(self, event: EventType, data: Optional[BasicEventHandlerData.BasicEventHandlerData]) -> None:
        nccl_log = MockNcclLog.get_instance()
        nccl_log.write_log(MockNcclLog.NcclLogLevel.DEBUG, "packet bundle call")

        if self.needs_processing:
            self.needs_processing = False
            self.delay = (self.generator.mem_write(self.size) + 
                         self.generator.mem_read(self.size) + 
                         self.generator.mem_read(self.size))
            self.generator.try_register_event(
                self,
                EventType.CommProcessingFinished,
                data,
                self.delay
            )
            return

        current = Sys.boosted_tick()

        if not Sys.PHY_MTP: 
            for packet in self.locked_packets:
                packet.ready_time = current

        if self.channel_id != -1 and self.flow_id is not None:
            ehd = BasicEventHandlerData(self.channel_id, self.flow_id)
        else:
            ehd = data  

        if self.channel_id == -1:
            self.stream.call(EventType.General, data)
        else:
            self.stream.call(EventType.NCCL_General, ehd)

