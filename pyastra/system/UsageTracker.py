from collections import namedtuple
# 假设存在Sys模块提供时间相关功能（需用户根据实际情况实现）
# 假设存在CSVWriter类（需用户根据实际情况实现）

# 对应C++中的Usage结构体/类
Usage = namedtuple('Usage', ['level', 'start', 'end'])

class UsageTracker:
    def __init__(self, levels: int):
        self.levels = levels
        self.current_level = 0
        self.last_tick = 0
        self.usage = []

    def increase_usage(self):
        """增加使用层级（不超过最大层级）"""
        if self.current_level < self.levels - 1:
            new_usage = Usage(
                level=self.current_level,
                start=self.last_tick,
                end=Sys.boostedTick()  # 需用户实现Sys.boostedTick()
            )
            self.usage.append(new_usage)
            self.current_level += 1
            self.last_tick = Sys.boostedTick()

    def decrease_usage(self):
        """减少使用层级（不低于0）"""
        if self.current_level > 0:
            new_usage = Usage(
                level=self.current_level,
                start=self.last_tick,
                end=Sys.boostedTick()
            )
            self.usage.append(new_usage)
            self.current_level -= 1
            self.last_tick = Sys.boostedTick()

    def set_usage(self, level: int):
        """设置目标使用层级（自动记录变更过程）"""
        if self.current_level != level:
            new_usage = Usage(
                level=self.current_level,
                start=self.last_tick,
                end=Sys.boostedTick()
            )
            self.usage.append(new_usage)
            self.current_level = level
            self.last_tick = Sys.boostedTick()

    def report(self, writer, offset: int):
        """将使用记录写入CSV文件"""
        col = offset * 3
        row = 1  # 行索引从1开始（假设CSV首行是表头）
        for entry in self.usage:
            writer.write_cell(row, col, str(entry.start))      # 写入开始时间
            writer.write_cell(row, col + 1, str(entry.level))  # 写入层级
            row += 1
        # writer.close_file_cont()  # 保留原注释（需用户实现关闭逻辑）

    def report_percentage(self, cycles: int) -> list[tuple[int, float]]:
        """计算每个周期内的使用百分比"""
        self.decrease_usage()  # 确保记录最后一个时间段
        self.increase_usage()  # 重新同步状态
        
        total_possible = (self.levels - 1) * cycles
        usage_ptr = 0  # 使用列表索引代替迭代器
        current_activity = 0
        period_start = 0
        period_end = cycles
        result = []

        while usage_ptr < len(self.usage):
            current_entry = self.usage[usage_ptr]
            # 计算当前记录与周期的交集
            begin = max(period_start, current_entry.start)
            end = min(period_end, current_entry.end)
            assert begin <= end, "时间区间错误"
            
            current_activity += (end - begin) * current_entry.level

            if current_entry.end >= period_end:
                # 完成当前周期计算
                percentage = (current_activity / total_possible) * 100
                result.append((period_end, percentage))
                # 重置周期
                period_start += cycles
                period_end += cycles
                current_activity = 0
            else:
                usage_ptr += 1

        return result

# 示例用法（需根据实际情况调整）
# class Sys:
#     @staticmethod
#     def boostedTick() -> int:
#         return int(time.time() * 1e6)  # 示例实现：返回微秒级时间戳

# class CSVWriter:
#     def write_cell(self, row: int, col: int, value: str):
#         # 实际实现应处理CSV写入逻辑
#         pass