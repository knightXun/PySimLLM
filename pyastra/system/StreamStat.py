# -*- coding: utf-8 -*-
from collections import deque

from ShareBusStat import *
from NetworkStat import *
from Common import * 

class StreamStat(SharedBusStat, NetworkStat):
    def __init__(self):
        super().__init__(BusType.Shared, 0, 0, 0, 0)
        self.queuing_delay = deque()  # 使用双端队列替代C++的std::list
        self.stream_stat_counter = 0

    def update_stream_stats(self, streamStat):
        self.update_bus_stats(BusType.Both, streamStat)
        self.update_network_stat(streamStat)

        if len(self.queuing_delay) < len(streamStat.queuing_delay):

            dif = len(streamStat.queuing_delay) - len(self.queuing_delay)
            for _ in range(dif):
                self.queuing_delay.append(0)

        it = iter(self.queuing_delay)
        for tick in streamStat.queuing_delay:
            next(it) += tick

        self.stream_stat_counter += 1

    def take_stream_stats_average(self):
        self.take_bus_stats_average()
        self.take_network_stat_average()

        if self.stream_stat_counter > 0:
            for i in range(len(self.queuing_delay)):
                self.queuing_delay[i] /= self.stream_stat_counter