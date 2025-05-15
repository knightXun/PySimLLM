from typing import Any, Callable, Optional
import time

from Sys import Sys
from BasicEventHandlerData import BasicEventHandlerData
from Common import EventType
from AstraNetworkAPI import MetaData, ncclFlowTag

class RecvPacketEventHandlerData(BasicEventHandlerData, MetaData):
    def __init__(self, owner, *args, **kwargs):
        if len(args) == 4 and all(isinstance(arg, int) for arg in args[:3]) and isinstance(args[3], int):
            nodeId, event, vnet, stream_num = args
            super().__init__(owner.owner, event)
            self.owner = owner
            self.vnet = vnet
            self.stream_num = stream_num
            self.message_end = True

            self.ready_time = Sys.boostedTick();
            self.flow_id = -2
            self.channel_id = kwargs.get('channel_id', 0) 
            self.child_flow_id = -1

        elif len(args) == 2 and isinstance(args[0], EventType) and isinstance(args[1], ncclFlowTag):
            event, flowTag = args
            super().__init__(owner.owner, event)
            self.flowTag = flowTag

        else:
            raise ValueError("Invalid constructor arguments")