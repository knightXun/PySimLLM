from abc import ABC, abstractmethod

from CallData import CallData
from Common import EventType


class Callable:
    def __del__(self):
        pass

    def call(self, type: EventType, data: CallData):
        pass

    