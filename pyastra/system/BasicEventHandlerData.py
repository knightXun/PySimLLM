
from CallData import CallData
from Sys import Sys
from Common import EventType

class BasicEventHandlerData(CallData):
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], Sys) and isinstance(args[1], EventType):
            self.node = args[0]
            self.event = args[1]
            self.channel_id = -1
            self.flow_id = -1
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            self.channel_id = args[0]
            self.flow_id = args[1]
        else:
            self.node = None
            self.event = None
            self.channel_id = None
            self.flow_id = None

    