import math

from Sys import Sys
from Common import ComType, InterDimensionScheduling
from AstraNetworkAPI import AstraNetworkAPI

class DimElapsedTime:
    def __init__(self, dim_num):
        self.dim_num = dim_num
        self.elapsed_time = 0

    def __lt__(self, other):
        return self.elapsed_time < other.elapsed_time


class OfflineGreedy:
    chunk_schedule = {}
    schedule_consumer = {}
    global_chunk_size = {}

    def __init__(self, sys):
        self.sys = sys
        if sys.dim_to_break == -1:
            self.dim_size = sys.physical_dims
            self.dim_BW = [0] * len(self.dim_size)
            for i in range(len(self.dim_size)):
                self.dim_BW[i] = sys.NI.get_BW_at_dimension(i)
                self.dim_elapsed_time.append(DimElapsedTime(i))
        else:
            self.dim_size = sys.logical_broken_dims
            self.dim_BW = [0] * len(self.dim_size)
            for i in range(len(self.dim_size)):
                if i > sys.dim_to_break:
                    self.dim_BW[i] = sys.NI.get_BW_at_dimension(i - 1)
                else:
                    self.dim_BW[i] = sys.NI.get_BW_at_dimension(i)
                self.dim_elapsed_time.append(DimElapsedTime(i))
        if sys.id == 0:
            print("Themis is configured with the following parameters: ")
            print("Dim size: ", end="")
            for i in range(len(self.dim_size)):
                print(self.dim_size[i], end=", ")
            print()
            print("BW per dim: ", end="")
            for i in range(len(self.dim_BW)):
                print(self.dim_BW[i], end=", ")
            print("\n")

    def get_chunk_size_from_elapsed_time(self, elapsed_time, dim, comm_type):
        if comm_type == ComType.Reduce_Scatter:
            result = ((elapsed_time * (self.dim_BW[dim.dim_num] / self.dim_BW[0])) / (
                    (dim.size[dim.dim_num] - 1) / dim.size[dim.dim_num])) * 1048576
            return result
        else:
            result = ((elapsed_time * (self.dim_BW[dim.dim_num] / self.dim_BW[0])) / (
                    (dim.size[dim.dim_num] - 1) / 1)) * 1048576
            return result

    def reset_loads(self):
        for i, dim in enumerate(self.dim_elapsed_time):
            dim.elapsed_time = 0
            dim.dim_num = i

    def get_chunk_scheduling(self, chunk_id, remaining_data_size, recommended_chunk_size, dimensions_involved,
                             inter_dim_scheduling, comm_type):
        if chunk_id in self.chunk_schedule:
            self.schedule_consumer[chunk_id] += 1
            if self.schedule_consumer[chunk_id] == len(self.sys.all_generators):
                res = self.chunk_schedule[chunk_id]
                remaining_data_size -= self.global_chunk_size[chunk_id]
                del self.chunk_schedule[chunk_id]
                del self.schedule_consumer[chunk_id]
                del self.global_chunk_size[chunk_id]
                return res
            remaining_data_size -= self.global_chunk_size[chunk_id]
            return self.chunk_schedule[chunk_id]
        if self.sys.id != 0:
            return self.sys.all_generators[0].offline_greedy.get_chunk_scheduling(
                chunk_id, remaining_data_size, recommended_chunk_size, dimensions_involved,
                inter_dim_scheduling, comm_type)
        else:
            if comm_type == ComType.All_Reduce:
                comm_type = ComType.Reduce_Scatter
            self.dim_elapsed_time.sort()
            if comm_type == ComType.All_Gather:
                self.dim_elapsed_time.reverse()
            result = []
            chunk_size = recommended_chunk_size
            chunk_size_calculated = False
            if inter_dim_scheduling == InterDimensionScheduling.OfflineGreedy:
                self.global_chunk_size[chunk_id] = min(remaining_data_size, chunk_size)
                remaining_data_size -= min(remaining_data_size, chunk_size)
            dim_elapsed_time_pointer = -1
            for dim in self.dim_elapsed_time:
                dim_elapsed_time_pointer += 1
                if not dimensions_involved[dim.dim_num] or self.dim_size[dim.dim_num] == 1:
                    result.append(dim.dim_num)
                    continue
                elif inter_dim_scheduling == InterDimensionScheduling.OfflineGreedyFlex and not chunk_size_calculated:
                    chunk_size_calculated = True
                    if comm_type == ComType.Reduce_Scatter:
                        load_difference = abs(self.dim_elapsed_time[-1].elapsed_time - dim.elapsed_time)
                        chunk_size = self.get_chunk_size_from_elapsed_time(load_difference, dim, ComType.Reduce_Scatter)
                    else:
                        lastIndex = len(self.dim_elapsed_time) - 1
                        while not dimensions_involved[self.dim_elapsed_time[lastIndex].dim_num] or \
                                self.dim_size[self.dim_elapsed_time[lastIndex].dim_num] == 1:
                            lastIndex -= 1
                        load_difference = abs(self.dim_elapsed_time[lastIndex].elapsed_time - dim.elapsed_time)
                        chunk_size = self.get_chunk_size_from_elapsed_time(load_difference,
                                                                           self.dim_elapsed_time[lastIndex],
                                                                           ComType.All_Gather)
                        lastIndex -= 1
                        while dim_elapsed_time_pointer <= lastIndex:
                            if dimensions_involved[self.dim_elapsed_time[lastIndex].dim_num] and \
                                    self.dim_size[self.dim_elapsed_time[lastIndex].dim_num] > 1:
                                chunk_size /= self.dim_size[self.dim_elapsed_time[lastIndex].dim_num]
                            lastIndex -= 1
                    if chunk_size < recommended_chunk_size:
                        result = list(range(len(self.dim_elapsed_time)))
                        self.global_chunk_size[chunk_id] = min(remaining_data_size, recommended_chunk_size)
                        chunk_size = min(remaining_data_size, recommended_chunk_size)
                        remaining_data_size -= min(remaining_data_size, recommended_chunk_size)
                        self.chunk_schedule[chunk_id] = result
                        self.schedule_consumer[chunk_id] = 1
                        myReordered = [self.dim_elapsed_time[0]] * len(self.dim_elapsed_time)
                        for myDim in range(len(self.dim_elapsed_time)):
                            for searchDim in range(len(self.dim_elapsed_time)):
                                if self.dim_elapsed_time[searchDim].dim_num == myDim:
                                    myReordered[myDim] = self.dim_elapsed_time[searchDim]
                                    break
                        self.dim_elapsed_time = myReordered
                        if comm_type == ComType.All_Gather:
                            self.dim_elapsed_time.reverse()
                        for myDim in range(len(self.dim_elapsed_time)):
                            if not dimensions_involved[myDim] or self.dim_size[myDim] == 1:
                                result.append(myDim)
                                continue
                            if comm_type == ComType.Reduce_Scatter:
                                self.dim_elapsed_time[myDim].elapsed_time += (
                                                                                     (chunk_size / 1048576) * (
                                                                                             (self.dim_size[myDim] - 1) /
                                                                                             self.dim_size[myDim])) / (
                                                                                     self.dim_BW[myDim] / self.dim_BW[0])
                                chunk_size /= self.dim_size[myDim]
                            else:
                                self.dim_elapsed_time[myDim].elapsed_time += (
                                                                                     (chunk_size / 1048576) * (
                                                                                             self.dim_size[myDim] - 1)) / (
                                                                                     self.dim_BW[myDim] / self.dim_BW[0])
                                chunk_size *= self.dim_size[myDim]
                        return result
                    else:
                        self.global_chunk_size[chunk_id] = min(remaining_data_size, chunk_size)
                        remaining_data_size -= min(remaining_data_size, chunk_size)
                elif inter_dim_scheduling == InterDimensionScheduling.OfflineGreedy and not chunk_size_calculated:
                    chunk_size_calculated = True
                    diff_size = 0
                    if comm_type == ComType.Reduce_Scatter:
                        load_difference = abs(self.dim_elapsed_time[-1].elapsed_time - dim.elapsed_time)
                        diff_size = self.get_chunk_size_from_elapsed_time(load_difference, dim, ComType.Reduce_Scatter)
                    else:
                        lastIndex = len(self.dim_elapsed_time) - 1
                        while not dimensions_involved[self.dim_elapsed_time[lastIndex].dim_num] or \
                                self.dim_size[self.dim_elapsed_time[lastIndex].dim_num] == 1:
                            lastIndex -= 1
                        load_difference = abs(self.dim_elapsed_time[lastIndex].elapsed_time - dim.elapsed_time)
                        diff_size = self.get_chunk_size_from_elapsed_time(load_difference,
                                                                          self.dim_elapsed_time[lastIndex],
                                                                          ComType.All_Gather)
                        lastIndex -= 1
                        while dim_elapsed_time_pointer <= lastIndex:
                            if dimensions_involved[self.dim_elapsed_time[lastIndex].dim_num] and \
                                    self.dim_size[self.dim_elapsed_time[lastIndex].dim_num] > 1:
                                diff_size /= self.dim_size[self.dim_elapsed_time[lastIndex].dim_num]
                            lastIndex -= 1
                    if diff_size < (recommended_chunk_size / 16):
                        result = list(range(len(self.dim_elapsed_time)))
                        self.chunk_schedule[chunk_id] = result
                        self.schedule_consumer[chunk_id] = 1
                        myReordered = [self.dim_elapsed_time[0]] * len(self.dim_elapsed_time)
                        for myDim in range(len(self.dim_elapsed_time)):
                            for searchDim in range(len(self.dim_elapsed_time)):
                                if self.dim_elapsed_time[searchDim].dim_num == myDim:
                                    myReordered[myDim] = self.dim_elapsed_time[searchDim]
                                    break
                        self.dim_elapsed_time = myReordered
                        if comm_type == ComType.All_Gather:
                            self.dim_elapsed_time.reverse()
                        for myDim in range(len(self.dim_elapsed_time)):
                            if not dimensions_involved[myDim] or self.dim_size[myDim] == 1:
                                continue
                            if comm_type == ComType.Reduce_Scatter:
                                self.dim_elapsed_time[myDim].elapsed_time += (
                                                                                     (chunk_size / 1048576) * (
                                                                                             (self.dim_size[myDim] - 1) /
                                                                                             self.dim_size[myDim])) / (
                                                                                     self.dim_BW[myDim] / self.dim_BW[0])
                                chunk_size /= self.dim_size[myDim]
                            else:
                                self.dim_elapsed_time[myDim].elapsed_time += (
                                                                                     (chunk_size / 1048576) * (
                                                                                             self.dim_size[myDim] - 1)) / (
                                                                                     self.dim_BW[myDim] / self.dim_BW[0])
                                chunk_size *= self.dim_size[myDim]
                        return result
                result.append(dim.dim_num)
                if comm_type == ComType.Reduce_Scatter:
                    dim.elapsed_time += ((chunk_size / 1048576) * (
                            (self.dim_size[dim.dim_num] - 1) / self.dim_size[dim.dim_num])) / (
                                                self.dim_BW[dim.dim_num] / self.dim_BW[0])
                    chunk_size /= self.dim_size[dim.dim_num]
                else:
                    dim.elapsed_time += ((chunk_size / 1048576) * (
                            self.dim_size[dim.dim_num] - 1)) / (
                                                self.dim_BW[dim.dim_num] / self.dim_BW[0])
                    chunk_size *= self.dim_size[dim.dim_num]
            self.chunk_schedule[chunk_id] = result
            self.schedule_consumer[chunk_id] = 1
            return result