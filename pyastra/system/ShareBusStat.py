# 假设已存在BusType枚举类型（需根据实际情况定义）
# 示例：
# from enum import Enum
# class BusType(Enum):
#     Shared = 1
#     Mem = 2
#     Other = 3

class CallData:
    """基础调用数据类（根据实际情况补充实现）"""
    pass

class SharedBusStat(CallData):
    def __init__(self, bus_type, total_bus_transfer_queue_delay, 
                 total_bus_transfer_delay, total_bus_processing_queue_delay, 
                 total_bus_processing_delay):
        """
        初始化共享总线统计类
        
        :param bus_type: 总线类型（BusType枚举）
        :param total_bus_transfer_queue_delay: 总线传输队列总延迟
        :param total_bus_transfer_delay: 总线传输总延迟
        :param total_bus_processing_queue_delay: 总线处理队列总延迟
        :param total_bus_processing_delay: 总线处理总延迟
        """
        # 共享总线相关统计
        self.total_shared_bus_transfer_queue_delay = 0.0
        self.total_shared_bus_transfer_delay = 0.0
        self.total_shared_bus_processing_queue_delay = 0.0
        self.total_shared_bus_processing_delay = 0.0

        # 内存总线相关统计
        self.total_mem_bus_transfer_queue_delay = 0.0
        self.total_mem_bus_transfer_delay = 0.0
        self.total_mem_bus_processing_queue_delay = 0.0
        self.total_mem_bus_processing_delay = 0.0

        # 初始化计数器
        self.shared_request_counter = 0
        self.mem_request_counter = 0

        # 根据总线类型初始化统计值
        if bus_type == BusType.Shared:
            self.total_shared_bus_transfer_queue_delay = total_bus_transfer_queue_delay
            self.total_shared_bus_transfer_delay = total_bus_transfer_delay
            self.total_shared_bus_processing_queue_delay = total_bus_processing_queue_delay
            self.total_shared_bus_processing_delay = total_bus_processing_delay
        elif bus_type == BusType.Mem:
            self.total_mem_bus_transfer_queue_delay = total_bus_transfer_queue_delay
            self.total_mem_bus_transfer_delay = total_bus_transfer_delay
            self.total_mem_bus_processing_queue_delay = total_bus_processing_queue_delay
            self.total_mem_bus_processing_delay = total_bus_processing_delay

    def update_bus_stats(self, bus_type, shared_bus_stat):
        """
        更新总线统计信息
        
        :param bus_type: 总线类型（BusType枚举）
        :param shared_bus_stat: 要合并的统计对象
        """
        if bus_type == BusType.Shared:
            self.total_shared_bus_transfer_queue_delay += shared_bus_stat.total_shared_bus_transfer_queue_delay
            self.total_shared_bus_transfer_delay += shared_bus_stat.total_shared_bus_transfer_delay
            self.total_shared_bus_processing_queue_delay += shared_bus_stat.total_shared_bus_processing_queue_delay
            self.total_shared_bus_processing_delay += shared_bus_stat.total_shared_bus_processing_delay
            self.shared_request_counter += 1
        elif bus_type == BusType.Mem:
            self.total_mem_bus_transfer_queue_delay += shared_bus_stat.total_mem_bus_transfer_queue_delay
            self.total_mem_bus_transfer_delay += shared_bus_stat.total_mem_bus_transfer_delay
            self.total_mem_bus_processing_queue_delay += shared_bus_stat.total_mem_bus_processing_queue_delay
            self.total_mem_bus_processing_delay += shared_bus_stat.total_mem_bus_processing_delay
            self.mem_request_counter += 1
        else:
            # 处理其他总线类型（同时更新两种总线统计）
            self.total_shared_bus_transfer_queue_delay += shared_bus_stat.total_shared_bus_transfer_queue_delay
            self.total_shared_bus_transfer_delay += shared_bus_stat.total_shared_bus_transfer_delay
            self.total_shared_bus_processing_queue_delay += shared_bus_stat.total_shared_bus_processing_queue_delay
            self.total_shared_bus_processing_delay += shared_bus_stat.total_shared_bus_processing_delay
            
            self.total_mem_bus_transfer_queue_delay += shared_bus_stat.total_mem_bus_transfer_queue_delay
            self.total_mem_bus_transfer_delay += shared_bus_stat.total_mem_bus_transfer_delay
            self.total_mem_bus_processing_queue_delay += shared_bus_stat.total_mem_bus_processing_queue_delay
            self.total_mem_bus_processing_delay += shared_bus_stat.total_mem_bus_processing_delay
            
            self.shared_request_counter += 1
            self.mem_request_counter += 1

    def take_bus_stats_average(self):
        """计算总线统计平均值（注意处理除零情况）"""
        # 计算共享总线平均值
        if self.shared_request_counter > 0:
            self.total_shared_bus_transfer_queue_delay /= self.shared_request_counter
            self.total_shared_bus_transfer_delay /= self.shared_request_counter
            self.total_shared_bus_processing_queue_delay /= self.shared_request_counter
            self.total_shared_bus_processing_delay /= self.shared_request_counter
        
        # 计算内存总线平均值
        if self.mem_request_counter > 0:
            self.total_mem_bus_transfer_queue_delay /= self.mem_request_counter
            self.total_mem_bus_transfer_delay /= self.mem_request_counter
            self.total_mem_bus_processing_queue_delay /= self.mem_request_counter
            self.total_mem_bus_processing_delay /= self.mem_request_counter