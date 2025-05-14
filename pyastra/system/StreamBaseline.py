# stream_baseline.py
from typing import List, Any, Optional
from BaseStream import BaseStream
from CollectivePhase import CollectivePhase
from Common import EventType, CallData, BusType
from RecvPacketEventHandlerData import RecvPacketEventHadndlerData
from SendPacketEventHandlerData import SendPacketEventHandlerData
from Sys import Sys
from DataSet import DataSet
from MockNcclLog import MockNcclLog
from CallData import *

class StreamBaseline(BaseStream):
    def __init__(
        self,
        owner: Sys,
        dataset: DataSet,
        stream_num: int,
        phases_to_go: List[CollectivePhase],
        priority: int
    ):
        super().__init__(stream_num, owner, phases_to_go)
        self.owner = owner
        self.stream_num = stream_num
        self.phases_to_go = phases_to_go
        self.dataset = dataset
        self.priority = priority
        self.steps_finished = 0
        self.initial_data_size = phases_to_go[0].initial_data_size if phases_to_go else 0

        # self.initialized = False
        # self.last_init: Optional[int] = None
        # self.last_phase_change: Optional[int] = None
        # self.queuing_delay: List[int] = []
        # self.total_packets_sent = 0
        # self.net_message_latency: List[int] = []
        # self.net_message_counter = 0

    def init(self) -> None:
        self.initialized = True
        self.last_init = Sys.boostedTick()
        if not self.my_current_phase.enabled:
            return
        
        self.my_current_phase.algorithm.run(EventType.StreamInit, None)
        
        nccl_log = MockNcclLog()
        nccl_log.write_log(MockNcclLog.NcclLogLevel.DEBUG, "StreamBaseline::algorithm->run finished")
        
        if self.steps_finished == 1:
            if self.last_phase_change is not None and self.creation_time is not None:
                self.queuing_delay.append(self.last_phase_change - self.creation_time)
        
        self.queuing_delay.append(Sys.boostedTick() - self.last_phase_change)
        self.total_packets_sent = 1

    def call(self, event: EventType, data: Optional[CallData]) -> None:
        if event == EventType.WaitForVnetTurn:
            self.owner.proceed_to_next_vnet_baseline(self)
            return
        elif event == EventType.NCCL_General:
            behd = data
            channel_id = behd.channel_id
            
            self.my_current_phase.algorithm.run(EventType.General, data)
        else:
            shared_bus_stat = data  
            self.update_bus_stats(BusType.Both, shared_bus_stat)

            self.my_current_phase.algorithm.run(EventType.General, data)
            if data is not None:
                del shared_bus_stat
                del data

    def consume(self, message: RecvPacketEventHadndlerData) -> None:       
        self.net_message_latency[-1] += Sys.boostedTick() - message.ready_time
        
        self.net_message_counter += 1
        
        self.my_current_phase.algorithm.run(EventType.PacketReceived, message)

    def sendcallback(self, messages: SendPacketEventHandlerData) -> None:
        if self.my_current_phase.algorithm is not None:
            self.my_current_phase.algorithm.run(EventType.PacketSentFinshed, messages)
    