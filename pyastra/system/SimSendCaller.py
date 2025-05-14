from Callable import Callable
from CallData import CallData

from Sys import Sys
import Common
from AstraNetworkAPI import sim_request

class SimSendCaller(Callable):    
    def __init__(
        self,
        generator: Sys,
        buffer,
        count: int,
        type: int,
        dst: int,
        tag: int,
        request: sim_request,
        msg_handler,
        fun_arg
    ) -> None:
        self.generator = generator
        self.buffer = buffer
        self.count = count
        self.type = type
        self.dst = dst
        self.tag = tag
        self.request = request
        self.msg_handler = msg_handler
        self.fun_arg = fun_arg

    def call(self, event_type: Common.EventType, data: CallData) -> None:
        self.generator.NI.sim_send(
            self.buffer,
            self.count,
            self.type,
            self.dst,
            self.tag,
            self.request,
            self.msg_handler,
            self.fun_arg
        )