from Callable import Callable
from Common import EventType
from CallData import CallData
from StreamStat import StreamStat
from MockNcclLog import MockNcclLog, NcclLogLevel
from Sys import Sys
from IntData import IntData

class DataSet(Callable, StreamStat):
    id_auto_increment = 0

    def __init__(self, total_streams):
        self.my_id = DataSet.id_auto_increment
        DataSet.id_auto_increment += 1
        self.total_streams = total_streams
        self.finished_streams = 0
        self.finished = False
        self.finish_tick = 0
        self.active = True
        self.creation_tick = Sys.boostedTick()
        self.notifier = None

    def set_notifier(self, layer, event):
        self.notifier = (layer, event)

    def notify_stream_finished(self, data):
        NcclLog = MockNcclLog.get_instance()
        NcclLog.write_log(NcclLogLevel.DEBUG, "notify_stream_finished id: %d finished_streams: %d total streams: %d notify %s",
                         self.my_id, self.finished_streams + 1, self.total_streams, self.notifier)
        self.finished_streams += 1
        if data is not None:
            self.update_stream_stats(data)
        if self.finished_streams == self.total_streams:
            self.finished = True
            self.finish_tick = Sys.boostedTick()
            if self.notifier is not None:
                NcclLog.write_log(NcclLogLevel.DEBUG, "notify_stream_finished notifier != nullptr ")
                self.take_stream_stats_average()
                c, ev = self.notifier
                self.notifier = None
                c.call(ev, IntData(self.my_id))
            else:
                NcclLog.write_log(NcclLogLevel.ERROR, "notify_stream_finished notifier = nullptr ")

    def call(self, event, data):
        self.notify_stream_finished(data)

    def is_finished(self):
        return self.finished

    def update_stream_stats(self, data):
        pass

    def take_stream_stats_average(self):
        pass    