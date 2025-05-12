from abc import ABC, abstractmethod


class CallData:
    pass


EventType = int


class Callable(ABC):
    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def call(self, type: EventType, data: CallData):
        pass

    