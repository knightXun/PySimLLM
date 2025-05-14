import os
import time
from enum import Enum
from system.Callable import *
from system.AstraSimDataAPI import *
from system.Sys import *


class ParallelismPolicy(Enum):
    MicroBenchmark = 0
    Data = 1
    Transformer = 2
    TransformerFwdInBckwd = 3
    DLRM = 4
    DLRMEnhanced = 5
    Model = 6
    HybridDataModel = 7
    HybridModelData = 8
    HybridCustomized = 9
    DistributedInference = 10
    All = 11
    None_ = 12

class LoopState(Enum):
    Forward_Pass = 0
    Weight_Gradient = 1
    Input_Gradient = 2
    Wait_For_Sim_Finish = 3
    Forward_In_BackPass = 4

class Workload(Callable):
    def __init__(self, run_name, generator, name, TOTAL_PASS, total_rows, 
        stat_row, path, seprate_log):
        
        self.initialized = False
        self.layers = []
        self.SIZE = 0
        self.counter = 0
        self.delay_loaded = False
        self.checkpoint_initiated = False
        self.collective_issued = False
        self.current_state = self.LoopState.Forward_Pass
        self.generator = generator
        self.TOTAL_PASS = TOTAL_PASS
        self.pass_counter = 0
        self.index = 0
        self.waiting_for_comm = 0
        self.detailed = None
        self.end_to_end = None
        self.dimension_utilization = None
        self.path = path
        self.stat_row = stat_row
        self.seprate_log = seprate_log
        self.initialized = self.initialize_workload(name)
        if not self.initialized:
            return
        self.total_rows = total_rows
        self.run_name = run_name
        self.registered_for_finished_streams = False
        if generator.id == 0 and seprate_log:
            print(f"stat path: {path} ,total rows: {total_rows} ,stat row: {stat_row}")
            self.detailed = CSVWriter(path, f"detailed_{generator.total_nodes}.csv")
            self.end_to_end = CSVWriter(path, "EndToEnd.csv")
            self.dimension_utilization = CSVWriter(path, f"{run_name}_dimension_utilization_{generator.npu_offset}.csv")
            if stat_row == 0:
                self.initialize_stat_files()

    def __del__(self):
        if self.end_to_end:
            del self.end_to_end
        if self.detailed:
            del self.detailed
        if self.dimension_utilization:
            del self.dimension_utilization
        for layer in self.layers:
            del layer
        self.layers = []

    def initialize_stat_files(self):
        self.detailed.initialize_csv(self.SIZE * self.total_rows + 20, 50)
        self.end_to_end.initialize_csv(self.SIZE * self.total_rows + 20, 50)

    def call(self, event, data):
        if self.counter > 0:
            if self.generator.id == 0:
                print("counter > 0")
            self.generator.try_register_event(self, EventType.Workload_Wait, None, self.counter)
            return
        if self.parallelismPolicy == ParallelismPolicy.Data:
            self.iterate_data_parallel()
        elif self.parallelismPolicy == ParallelismPolicy.Transformer:
            self.iterate_hybrid_parallel_Transformer()
        elif self.parallelismPolicy in [ParallelismPolicy.DLRM, ParallelismPolicy.DLRMEnhanced]:
            self.iterate_hybrid_parallel_DLRM()
        elif self.parallelismPolicy == ParallelismPolicy.MicroBenchmark:
            self.iterate_micro_benchmark()
        elif self.parallelismPolicy == ParallelismPolicy.Model:
            self.iterate_model_parallel()
        elif self.parallelismPolicy == ParallelismPolicy.HybridDataModel:
            self.iterate_hybrid_parallel_data_model()
        elif self.parallelismPolicy == ParallelismPolicy.HybridModelData:
            self.iterate_hybrid_parallel_model_data()
        elif self.parallelismPolicy == ParallelismPolicy.DistributedInference:
            self.iterate_distributed_inference()
        elif self.parallelismPolicy == ParallelismPolicy.TransformerFwdInBckwd:
            self.iterate_hybrid_parallel_Transformer_fwd_in_bckwd()
        elif self.parallelismPolicy == ParallelismPolicy.HybridCustomized:
            self.iterate_hybrid_parallel_customized()
        else:
            raise ValueError("No known parallelism!")

    def report(self):
        total_compute = 0
        total_exposed = 0
        pre_bubble_time = 0
        DP_comm = 0
        DP_EP_comm = 0
        Expose_TP_comm = 0
        Expose_EP_comm = 0
        total_fwd_time = [0, 0, 0]
        total_wg_time = [0, 0, 0]
        total_ig_time = [0, 0, 0]
        astraSimDataAPI = {
            "run_name": self.run_name,
            "workload_finished_time": self.generator.boostedTick() / 1000,
            "layers_stats": []
        }
        print(f"workload stats for the job scheduled at NPU offset: {self.generator.npu_offset}")
        for i in range(self.SIZE):
            layer_stats = self.layers[i].report(
                self.run_name,
                i,
                self.total_rows,
                self.stat_row,
                self.detailed,
                self.end_to_end,
                total_compute,
                total_exposed,
                self.seprate_log,
                total_fwd_time,
                total_wg_time,
                total_ig_time,
                pre_bubble_time,
                DP_comm,
                DP_EP_comm,
                Expose_TP_comm,
                Expose_EP_comm
            )
            astraSimDataAPI["layers_stats"].append(layer_stats)
        astraSimDataAPI["total_compute"] = total_compute
        astraSimDataAPI["total_exposed_comm"] = total_exposed
        astraSimDataAPI["avg_chunk_latency_per_logical_dimension"] = [
            latency / 1000 for latency in self.generator.scheduler_unit.get_average_latency_per_dimension()
        ]
        print("*************************")
        print(f"all passes finished at time: {self.generator.boostedTick()}, id of first layer: {self.layers[0].id}")
        self.generator.NI.pass_front_end_report(astraSimDataAPI)
        if self.seprate_log:
            dims = []
            for i in range(len(self.generator.scheduler_unit.usage)):
                dims.append(self.generator.scheduler_unit.usage[i].report_percentage(10000))
            self.dimension_utilization.finalize_csv(dims)

    def check_for_sim_end(self):
        if self.pass_counter == self.TOTAL_PASS:
            self.current_state = self.LoopState.Wait_For_Sim_Finish
            if self.generator.streams_finished != self.generator.streams_injected and not self.registered_for_finished_streams:
                self.generator.register_for_finished_stream(self)
                self.registered_for_finished_streams = True
                self.layers[0].is_weight_grad_comm_finished_blocking()
                return
            if self.generator.streams_finished == self.generator.streams_injected:
                if self.generator.id == 0:
                    self.report()
                self.generator.workload_finished()
                return
        return

    def iterate_micro_benchmark(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        if self.current_state != self.LoopState.Wait_For_Sim_Finish:
            for _ in range(self.TOTAL_PASS):
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Non_Blocking
                )
        self.check_for_sim_end()

    def iterate_data_parallel(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            self.index += 1
            self.delay_loaded = False
            if self.index >= self.SIZE:
                self.current_state = self.LoopState.Weight_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Weight_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_weight_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            self.delay_loaded = False
            self.layers[self.index].issue_weight_grad_comm(
                SchedulingPolicy.None_, CollectiveBarrier.Non_Blocking
            )
            if self.index == 0:
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = self.LoopState.Forward_Pass
            else:
                self.current_state = self.LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Input_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            self.delay_loaded = False
            self.index -= 1
            self.current_state = self.LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_customized(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = self.LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Weight_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_weight_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.FIFO, CollectiveBarrier.Non_Blocking
                )
            if not self.layers[self.index].is_input_grad_comm_finished_blocking():
                return
            self.collective_issued = False
            self.delay_loaded = False
            if self.index >= 0:
                self.index -= 1
            if self.index == -1:
                self.index = 0
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = self.LoopState.Forward_Pass
            else:
                self.current_state = self.LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Input_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued and self.index > 0:
                self.collective_issued = True
                self.layers[self.index].issue_input_grad_comm(
                    SchedulingPolicy.LIFO, CollectiveBarrier.Non_Blocking
                )
            self.collective_issued = False
            self.delay_loaded = False
            self.current_state = self.LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_data_model(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = self.LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Weight_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_weight_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.FIFO, CollectiveBarrier.Non_Blocking
                )
            if not self.layers[self.index].is_input_grad_comm_finished_blocking():
                return
            self.collective_issued = False
            self.delay_loaded = False
            if self.index >= 0:
                self.index -= 1
            if self.index == -1:
                self.index = 0
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = self.LoopState.Forward_Pass
            else:
                self.current_state = self.LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Input_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued and self.index > 0:
                self.collective_issued = True
                self.layers[self.index].issue_input_grad_comm(
                    SchedulingPolicy.LIFO, CollectiveBarrier.Non_Blocking
                )
            self.collective_issued = False
            self.delay_loaded = False
            self.current_state = self.LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_model_data(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = self.LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Weight_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_weight_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.FIFO, CollectiveBarrier.Non_Blocking
                )
            if not self.layers[self.index].is_input_grad_comm_finished_blocking():
                return
            self.collective_issued = False
            self.delay_loaded = False
            if self.index >= 0:
                self.index -= 1
            if self.index == -1:
                self.index = 0
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = self.LoopState.Forward_Pass
            else:
                self.current_state = self.LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Input_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued and self.index > 0:
                self.collective_issued = True
                self.layers[self.index].issue_input_grad_comm(
                    SchedulingPolicy.LIFO, CollectiveBarrier.Non_Blocking
                )
            self.collective_issued = False
            self.delay_loaded = False
            self.current_state = self.LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_distributed_inference(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.index = 0
                self.pass_counter += 1
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_model_parallel(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                involved_dimensions = [True, True, True]
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = self.LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Weight_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_weight_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.layers[self.index].is_input_grad_comm_finished_blocking():
                return
            self.collective_issued = False
            self.delay_loaded = False
            if self.index >= 0:
                self.index -= 1
            if self.index == -1:
                self.index = 0
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = self.LoopState.Forward_Pass
            else:
                self.current_state = self.LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Input_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued and self.index > 0:
                self.collective_issued = True
                involved_dimensions = [True, True, True]
                self.layers[self.index].issue_input_grad_comm(
                    SchedulingPolicy.LIFO, CollectiveBarrier.Non_Blocking
                )
            self.collective_issued = False
            self.delay_loaded = False
            self.current_state = self.LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_Transformer(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.None_, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = self.LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Weight_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_weight_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.FIFO, CollectiveBarrier.Non_Blocking
                )
            if not self.layers[self.index].is_input_grad_comm_finished_blocking():
                return
            self.collective_issued = False
            self.delay_loaded = False
            if self.index >= 0:
                self.index -= 1
            if self.index == -1:
                self.index = 0
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = self.LoopState.Forward_Pass
            else:
                self.current_state = self.LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == self.LoopState.Input_Gradient:
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                self.layers[self.index].issue_input_grad_comm(
                    SchedulingPolicy.LIFO, CollectiveBarrier.Blocking
                )
                return
            self.collective_issued = False
            self.delay_loaded = False
            self.current_state = self.LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_Transformer_fwd_in_bckwd(self):
        NcclLog = MockNcclLog.getInstance()
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == self.LoopState.Forward_Pass:
            if not self.layers[self.index].is_weight_grad_comm_finished_blocking():
                return
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_fwd_pass_compute()
                self.delay_loaded = True
            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return
            if not self.collective_issued:
                self.collective_issued = True
                if 0 < self.layers[self.index].fwd_pass_comm_size 