from Callable import Callable
from CallData import CallData

from Sys import Sys
import Common
from AstraNetworkAPI import sim_request

class SimRecvCaller(Callable):
    def __init__(
        self,
        generator,
        buffer,
        count: int,
        type: int,
        src: int,
        tag: int,
        request: sim_request, 
        msg_handler,
        fun_arg
    ):
        self.generator = generator
        self.buffer = buffer
        self.count = count
        self.type = type
        self.src = src
        self.tag = tag
        self.request = request
        self.msg_handler = msg_handler
        self.fun_arg = fun_arg

    def call(self, type: int, data: CallData) -> None:  
        self.generator.NI.sim_recv(
            self.buffer,
            self.count,
            self.type,
            self.src,
            self.tag,
            self.request,
            self.msg_handler,
            self.fun_arg
        )