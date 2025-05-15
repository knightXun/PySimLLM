from typing import Any, Callable, Optional

from BasicEventHandlerData import BasicEventHandlerData
from Common import EventType
from SimSendCaller import SimSendCaller
from AstraNetworkAPI import MetaData

class RendezvousSendData(BasicEventHandlerData, MetaData):
    
    def __init__(
        self,
        node_id: int,
        generator: Any,
        buffer: Any,
        count: int,
        type_: int,  
        dst: int,
        tag: int,
        request: Any,
        msg_handler: Optional[Callable[[Any], None]] = None,
        fun_arg: Optional[Any] = None
    ) -> None:
        super().__init__(generator, EventType.RendezvousSend)
        
        self.send = SimSendCaller(
            generator=generator,
            buffer=buffer,
            count=count,
            type_=type_,
            dst=dst,
            tag=tag,
            request=request,
            msg_handler=msg_handler,
            fun_arg=fun_arg
        )