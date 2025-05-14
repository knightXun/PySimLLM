from collections import namedtuple

import Sys
from Callable import *
from CallData import * 
from Common import *
from Usage import *
from workload.CSVWriter import *


class UsageTracker:
    def __init__(self, levels: int):
        self.levels = levels
        self.current_level = 0
        self.last_tick = 0
        self.usage = []

    def increase_usage(self):
        if self.current_level < self.levels - 1:
            new_usage = Usage(
                level=self.current_level,
                start=self.last_tick,
                end= Sys.boostedTick() 
            )

            self.usage.append(new_usage)
            self.current_level += 1
            self.last_tick = Sys.boostedTick()

    def decrease_usage(self):
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
        col = offset * 3
        row = 1  
        for entry in self.usage:
            writer.write_cell(row, col, str(entry.start))     
            writer.write_cell(row, col + 1, str(entry.level)) 
            row += 1
        # writer.close_file_cont()  

    def report_percentage(self, cycles: int) -> list[tuple[int, float]]:
        self.decrease_usage()  
        self.increase_usage() 
        
        total_possible = (self.levels - 1) * cycles
        usage_ptr = 0  
        current_activity = 0
        period_start = 0
        period_end = cycles
        result = []

        while usage_ptr < len(self.usage):
            current_entry = self.usage[usage_ptr]
            begin = max(period_start, current_entry.start)
            end = min(period_end, current_entry.end)
            assert begin <= end, "时间区间错误"
            
            current_activity += (end - begin) * current_entry.level

            if current_entry.end >= period_end:
                percentage = (current_activity / total_possible) * 100
                result.append((period_end, percentage))
                period_start += cycles
                period_end += cycles
                current_activity = 0
            else:
                usage_ptr += 1

        return result