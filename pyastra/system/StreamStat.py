# -*- coding: utf-8 -*-
from collections import deque

class StreamStat(SharedBusStat, NetworkStat):
    def __init__(self):
        # 调用父类构造函数（假设SharedBusStat需要BusType参数）
        super().__init__(BusType.Shared, 0, 0, 0, 0)
        self.queuing_delay = deque()  # 使用双端队列替代C++的std::list
        self.stream_stat_counter = 0

    def update_stream_stats(self, stream_stat):
        """更新流统计数据"""
        # 更新总线统计和网络统计（假设父类已实现对应方法）
        self.update_bus_stats(BusType.Both, stream_stat)
        self.update_network_stat(stream_stat)

        # 扩展当前队列长度以匹配传入对象的队列长度
        if len(self.queuing_delay) < len(stream_stat.queuing_delay):
            length_diff = len(stream_stat.queuing_delay) - len(self.queuing_delay)
            self.queuing_delay.extend([0.0] * length_diff)

        # 累加队列延迟数据
        for i, tick in enumerate(stream_stat.queuing_delay):
            self.queuing_delay[i] += tick

        self.stream_stat_counter += 1

    def take_stream_stats_average(self):
        """计算流统计数据的平均值"""
        # 调用父类方法计算总线和网络统计的平均值（假设父类已实现）
        self.take_bus_stats_average()
        self.take_network_stat_average()

        # 计算队列延迟的平均值
        if self.stream_stat_counter > 0:
            for i in range(len(self.queuing_delay)):
                self.queuing_delay[i] /= self.stream_stat_counter