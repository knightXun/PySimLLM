import os
import time
from enum import Enum
from system.Callable import *
from system.AstraSimDataAPI import *
from system.Sys import *
from system.MockNcclGroup import GroupType
from Layer import Layer 

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
    NONE = 12

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
        self.run_type = ""
        self.counter = 0
        self.delay_loaded = False
        self.checkpoint_initiated = False
        self.collective_issued = False
        self.current_state = LoopState.Forward_Pass
        self.generator = generator
        self.TOTAL_PASS = TOTAL_PASS
        self.DLRM_LAST_BOTTOM_LAYER = 0
        self.pass_counter = 0
        self.pending_collectives = 0
        self.model_parallel_npu_group = 0  # tp size 
        self.expert_parallel_npu_group = 0  # ep size 
        self.pipeline_model_parallelism = 0 # pp size 
        self.index = 0 
        self.GA = 0 # GA size
        self.all_gpus = 0
        self.vpp = 0
        self.pp_commsize = 0
        self.waiting_for_comm = 0
        self.detailed = None
        self.end_to_end = None
        self.dimension_utilization = None
        self.path = path
        self.stat_row = stat_row
        self.seprate_log = seprate_log
        self.parallelismPolicy = ParallelismPolicy.NONE
        
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
        # if self.end_to_end:
        #     del self.end_to_end
        # if self.detailed:
        #     del self.detailed
        # if self.dimension_utilization:
        #     del self.dimension_utilization
        # for layer in self.layers:
        #     del layer
        # self.layers = []
        pass

    def initialize_stat_files(self):
        self.detailed.initialize_csv(self.SIZE * self.total_rows + 20, 50)
        self.end_to_end.initialize_csv(self.SIZE * self.total_rows + 20, 50)
        # #ifdef NS3_MPI
        # detailed->initialize_csv(SIZE * total_rows + 20, 50);
        # #endif
        # #ifdef NS3_MTP 
        # detailed->initialize_csv(SIZE * total_rows + 20, 50);
        # #endif
        # end_to_end->initialize_csv(SIZE * total_rows + 20, 50);

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
        total_compute = 0.0
        total_exposed = 0.0
        pre_bubble_time = 0.0
        DP_comm = 0.0
        DP_EP_comm = 0.0
        Expose_TP_comm = 0.0
        Expose_EP_comm = 0.0
        total_fwd_time = [0.0, 0.0, 0.0]
        total_wg_time = [0.0, 0.0, 0.0]
        total_ig_time = [0.0, 0.0, 0.0]
        astraSimDataAPI = AstraSimDataAPI()
        astraSimDataAPI.run_name = self.run_name
        astraSimDataAPI.workload_finished_time = Sys.boostedTick() / 1000
        astraSimDataAPI.layers_stats = []

        print(f"workload stats for the job scheduled at NPU offset: {self.generator.npu_offset}")
        for i in range(self.SIZE):
            #TODO: #ifdef ANALYTI 这里逻辑还有欠缺，后面补齐，源代码存在宏定义
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
        
            astraSimDataAPI.layers_stats.append(layer_stats)

        astraSimDataAPI.total_compute = total_compute
        astraSimDataAPI.total_exposed_comm = total_exposed
        astraSimDataAPI.avg_chunk_latency_per_logical_dimension = [
            latency / Common.FREQ for latency in self.generator.scheduler_unit.get_average_latency_per_dimension()
        ]
        print("*************************")
        print(f"all passes finished at time: {self.generator.boostedTick()}, id of first layer: {self.layers[0].id}")
        self.generator.NI.pass_front_end_report(astraSimDataAPI)
        if self.seprate_log:
            dims = []
            for i in range(len(self.generator.scheduler_unit.usage)):
                dims.append(self.generator.scheduler_unit.usage[i].report_percentage(10000))
            self.dimension_utilization.finalize_csv(dims)

        # #ifdef NS3_MTP 
        # if (this->seprate_log) {
        #     std::list<std::list<std::pair<uint64_t, double>>> dims;
        #     for (int i = 0; i < generator->scheduler_unit->usage.size(); i++) {
        #     dims.push_back(
        #         generator->scheduler_unit->usage[i].report_percentage(10000));
        #     }
        #     dimension_utilization->finalize_csv(dims);
        # }
        # #endif
        # #ifdef NS3_MPI 
        # if (this->seprate_log) {
        #     std::list<std::list<std::pair<uint64_t, double>>> dims;
        #     for (int i = 0; i < generator->scheduler_unit->usage.size(); i++) {
        #     dims.push_back(
        #         generator->scheduler_unit->usage[i].report_percentage(10000));
        #     }
        #     dimension_utilization->finalize_csv(dims);
        # }
        # #endif

    def check_for_sim_end(self):
        if self.pass_counter == self.TOTAL_PASS:
            self.current_state = LoopState.Wait_For_Sim_Finish
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
        if self.current_state != LoopState.Wait_For_Sim_Finish:
            for _ in range(self.TOTAL_PASS):
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.NONE, CollectiveBarrier.Non_Blocking
                )
        self.check_for_sim_end()

    def iterate_data_parallel(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == LoopState.Forward_Pass:
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
                self.current_state = LoopState.Weight_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Weight_Gradient:
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
                SchedulingPolicy.NONE, CollectiveBarrier.Non_Blocking
            )
            if self.index == 0:
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {self.generator.boostedTick()}")
                self.pass_counter += 1
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Input_Gradient:
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
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_customized(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == LoopState.Forward_Pass:
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Weight_Gradient:
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
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Input_Gradient:
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
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_data_model(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == LoopState.Forward_Pass:
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Weight_Gradient:
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
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Input_Gradient:
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
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_model_data(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == LoopState.Forward_Pass:
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Weight_Gradient:
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
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Input_Gradient:
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
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_distributed_inference(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == LoopState.Forward_Pass:
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
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
        if self.current_state == LoopState.Forward_Pass:
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Weight_Gradient:
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
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Input_Gradient:
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
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_Transformer(self):
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()
        if self.current_state == LoopState.Forward_Pass:
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False
            if self.index >= self.SIZE:
                self.current_state = LoopState.Input_Gradient
                self.index -= 1
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Weight_Gradient:
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
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return
        elif self.current_state == LoopState.Input_Gradient:
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
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def iterate_hybrid_parallel_Transformer_fwd_in_bckwd(self):
        """迭代混合并行Transformer的前向传播和反向传播过程"""
        
        logger = MockNcclLog.get_instance()

        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()

        if self.current_state == LoopState.Forward_Pass:
            # 前向传播阶段
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
                # 确保通信大小不小于4096
                if (self.layers[self.index].fwd_pass_comm_size < 4096 and 
                    self.layers[self.index].fwd_pass_comm_size > 0):
                    self.layers[self.index].fwd_pass_comm_size = 4096
                # 发起前向通信
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return

            # 移动到下一层
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False

            if self.index >= self.SIZE:
                self.current_state = LoopState.Input_Gradient
                self.index -= 1

            logger.write_log("DEBUG", "workload::call fwd_pass register_event EventType::General")
            self.generator.register_event(self, EventType.General, None, 1)
            return

        elif self.current_state == LoopState.Weight_Gradient:
            # 权重梯度计算阶段
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
                # 发起权重梯度通信
                self.layers[self.index].issue_weight_grad_comm(
                    SchedulingPolicy.FIFO, CollectiveBarrier.Non_Blocking
                )

            # 等待输入梯度通信完成
            if not self.layers[self.index].is_input_grad_comm_finished_blocking():
                return

            self.collective_issued = False
            self.delay_loaded = False

            if self.index >= 0:
                self.index -= 1

            # 检查是否完成一个完整的训练周期
            if self.index == -1:
                self.index = 0
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {Sys.boostedTick()}")
                self.pass_counter += 1
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient

            self.generator.register_event(self, EventType.General, None, 1)
            return

        elif self.current_state == LoopState.Input_Gradient:
            # 输入梯度计算阶段
            # 检查是否需要在反向传播中重新执行前向传播
            if (self.layers[self.index].needs_fwd_in_bckwd_initiation and 
                not self.checkpoint_initiated):
                tmp = self.index
                # 找到最近的检查点层
                while not self.layers[self.index].is_checkpoint:
                    self.index -= 1
                self.index += 1
                self.current_state = LoopState.Forward_In_BackPass
                self.checkpoint_initiated = True
                self.generator.register_event(self, EventType.General, None, 1)
                if self.generator.id == 0:
                    print(f"***** info, initiating fwd_in_bkwd starting from layer:{self.index} to layer: {tmp} ,at time: {Sys.boostedTick()}")
                return

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
                # 发起输入梯度通信
                self.layers[self.index].issue_input_grad_comm(
                    SchedulingPolicy.LIFO, CollectiveBarrier.Blocking
                )
                return

            self.checkpoint_initiated = False
            self.collective_issued = False
            self.delay_loaded = False
            self.current_state = LoopState.Weight_Gradient
            self.generator.register_event(self, EventType.General, None, 1)
            return

        elif self.current_state == LoopState.Forward_In_BackPass:
            # 在反向传播中执行前向传播
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
                # 发起前向通信
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.NONE, CollectiveBarrier.Blocking
                )
                return

            # 移动到下一层
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False

            # 检查是否需要切换到输入梯度阶段
            if self.layers[self.index].needs_fwd_in_bckwd_initiation:
                self.current_state = LoopState.Input_Gradient

            self.generator.register_event(self, EventType.General, None, 1)
            return


    def iterate_hybrid_parallel_DLRM(self):
        """迭代执行混合并行DLRM模型的训练过程"""
        assert self.index >= 0
        assert self.index < self.SIZE
        self.check_for_sim_end()

        if self.current_state == LoopState.Forward_Pass:
            # 前向传播阶段
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

            # 处理All_to_All通信或特定层的依赖
            if (not self.collective_issued and
                self.layers[self.index].fwd_pass_comm_type == ComType.All_to_All):
                self.collective_issued = True
                self.layers[self.index].issue_forward_pass_comm(
                    SchedulingPolicy.HIGHEST, CollectiveBarrier.Non_Blocking
                )
            elif self.index == self.DLRM_LAST_BOTTOM_LAYER:
                if not self.layers[0].is_fwd_pass_comm_finished_blocking():
                    return

            # 移动到下一层
            self.index += 1
            self.delay_loaded = False
            self.collective_issued = False

            if self.index >= self.SIZE:
                self.current_state = LoopState.Weight_Gradient
                self.index -= 1

            if self.generator.id == 0:
                print(f"*************************layer changed to: {self.index}")

            self.generator.register_event(self, EventType.General, None, 1)
            return

        elif self.current_state == LoopState.Weight_Gradient:
            # 权重梯度计算阶段
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
                    SchedulingPolicy.NONE, CollectiveBarrier.Non_Blocking
                )

            # 检查输入梯度通信是否完成
            if (self.parallelismPolicy == ParallelismPolicy.DLRM and
                not self.layers[self.index].is_input_grad_comm_finished_blocking()):
                return

            # 完成一个完整的训练周期
            if self.index == 0:
                if self.generator.id == 0:
                    print(f"pass: {self.pass_counter} finished at time: {Sys.boostedTick()}")
                self.pass_counter += 1
                self.current_state = LoopState.Forward_Pass
            else:
                self.current_state = LoopState.Input_Gradient

            self.delay_loaded = False
            self.collective_issued = False
            self.generator.register_event(self, EventType.General, None, 1)
            return

        elif self.current_state == LoopState.Input_Gradient:
            # 输入梯度计算阶段
            if not self.delay_loaded:
                self.counter = self.layers[self.index].get_input_grad_compute()
                self.delay_loaded = True

            if self.counter > 0:
                self.generator.try_register_event(
                    self, EventType.Workload_Wait, None, self.counter
                )
                return

            # 特殊处理特定层的输入梯度通信
            if self.index == self.DLRM_LAST_BOTTOM_LAYER + 1:
                self.layers[0].issue_input_grad_comm(
                    SchedulingPolicy.HIGHEST, CollectiveBarrier.Non_Blocking
                )

            # 移动到上一层
            self.index -= 1
            if self.generator.id == 0:
                print(f"*************************layer changed to: {self.index} in ig")

            self.current_state = LoopState.Weight_Gradient
            self.collective_issued = False
            self.delay_loaded = False
            self.generator.register_event(self, EventType.General, None, 1)
            return

    def get_layer_numbers(self, workload_input: str) -> int:
        try:
            with open(f"workload_inputs/{workload_input}", 'r') as inFile:
                print("Success in opening workload file")
                # 跳过第一行
                inFile.readline()
                # 读取层数
                layers = int(inFile.readline().strip())
                return layers
        except FileNotFoundError:
            print(f"Unable to open file: {workload_input}")
            print("This error is fatal. Please check your path and filename.")
            exit(1)

    def decode_parallelsim(self, parallelism: str) -> ParallelismPolicy:
        mapping = {
            "DATA": ParallelismPolicy.Data,
            "HYBRID_TRANSFORMER": ParallelismPolicy.Transformer,
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD": ParallelismPolicy.TransformerFwdInBckwd,
            "HYBRID_DLRM": ParallelismPolicy.DLRM,
            "HYBRID_DLRM_ENHANCED": ParallelismPolicy.DLRMEnhanced,
            "MODEL": ParallelismPolicy.Model,
            "HYBRID_DATA_MODEL": ParallelismPolicy.HybridDataModel,
            "HYBRID_MODEL_DATA": ParallelismPolicy.HybridModelData,
            "HYBRID_CUSTOMIZED": ParallelismPolicy.HybridCustomized,
            "MICRO": ParallelismPolicy.MicroBenchmark,
            "DISTRIBUTED_INFERENCE": ParallelismPolicy.DistributedInference,
        }
        return mapping.get(parallelism, ParallelismPolicy.NONE)

    def decode_involved_dimensions(
        self, policy: ParallelismPolicy, model_parallel_npu_group: int
    ) -> dict[str, list[bool]]:
        """解析维度"""
        none = [False] * 10
        all_ = [True] * 10
        
        if policy in (ParallelismPolicy.All,):
            return {"fwd": all_, "ig": all_, "wg": all_}
        elif policy in (ParallelismPolicy.Data, ParallelismPolicy.DLRM, 
                        ParallelismPolicy.DLRMEnhanced, ParallelismPolicy.MicroBenchmark):
            return {"fwd": none, "ig": none, "wg": all_}
        elif policy in (ParallelismPolicy.Model, ParallelismPolicy.DistributedInference):
            return {"fwd": all_, "ig": all_, "wg": none}
        elif policy == ParallelismPolicy.HybridModelData:
            data = [True, False] + none[2:]
            model = [False, True] + all_[2:]
            return {"fwd": model, "ig": model, "wg": data}
        elif policy == ParallelismPolicy.HybridDataModel:
            model = [True, False] + none[2:]
            data = [False, True] + all_[2:]
            return {"fwd": model, "ig": model, "wg": data}
        elif policy in (ParallelismPolicy.Transformer, ParallelismPolicy.TransformerFwdInBckwd):
            model_parallel_boundary = self.generator.break_dimension(model_parallel_npu_group)
            model = [True] * (model_parallel_boundary + 1) + [False] * (9 - model_parallel_boundary)
            data = [False] * (model_parallel_boundary + 1) + [True] * (9 - model_parallel_boundary)
            return {"fwd": model, "ig": model, "wg": data}
        else:
            return {"fwd": none, "ig": none, "wg": none}

    def initialize_workload(self, name: str) -> bool:
        """初始化工作负载配置（部分实现）"""
        checkpoints = {}  
        need_checkpoint_initiation = {}  
        tokens = []
        try:
            with open(name, 'r') as inFile:
                # 处理文件打开成功的情况
                if self.generator.id == 0:
                    print("Success in opening workload file")

                # 读取第一行并分词
                firstline = inFile.readline().strip()
                tokens = firstline.split()

                # 解析并行策略
                if tokens:
                    self.parallelismPolicy = self.decode_parallelsim(tokens[0])
                else:
                    raise ValueError("First line of workload file is empty")

            # 解析Transformer/TransformerFwdInBckwd策略的参数
            if self.parallelismPolicy in {ParallelismPolicy.Transformer, ParallelismPolicy.TransformerFwdInBckwd}:
                # 解析通用参数
                for i in range(1, len(tokens)):
                    if tokens[i] == "model_parallel_NPU_group:":
                        self.model_parallel_npu_group = int(tokens[i+1])
                        if self.generator.id == 0:
                            print(f"model_parallel_NPU_group is {self.model_parallel_npu_group}")
                    elif tokens[i] == "ep:":
                        self.expert_parallel_npu_group = int(tokens[i+1])
                    elif tokens[i] == "pp:":
                        self.pipeline_model_parallelism = int(tokens[i+1])
                    elif tokens[i] == "vpp:":
                        self.vpp = int(tokens[i+1])
                    elif tokens[i] == "ga:":
                        self.GA = int(tokens[i+1])
                    elif tokens[i] == "all_gpus:":
                        self.all_gpus = int(tokens[i+1])
                
                # 解析TransformerFwdInBckwd特有的检查点信息
                if self.parallelismPolicy == ParallelismPolicy.TransformerFwdInBckwd:
                    if self.generator.id == 0:
                        print("checkpoints layers are: ")
                    
                    i = 1
                    while i < len(tokens):
                        if tokens[i] == "checkpoints:":
                            count = int(tokens[i+1])
                            while count > 0:
                                count = count - 1
                                layer = int(tokens[i+2])
                                checkpoints[layer] = True
                                if self.generator.id == 0:
                                    print(f"{layer}, ")
                        elif tokens[i] == "checkpoint_initiates:":
                            if self.generator.id == 0:
                                print()
                                print("layers initiating fwd_in_bckwd are: ")
                        
                            count = int(tokens[i+1])
                            while count > 0:
                                count = count - 1
                                layer = int(tokens[i+2])
                                need_checkpoint_initiation[layer] = True
                                if self.generator.id == 0:
                                    print(layer, ", ")

                            if self.generator.id == 0:
                                print()
                        else:
                            i += 1
                    if self.generator.id == 0 and "checkpoint_initiates:" in tokens:
                        print()
            # 解析DLRM/DLRMEnhanced策略的参数
            elif self.parallelismPolicy in {ParallelismPolicy.DLRM, ParallelismPolicy.DLRMEnhanced}:
                for i in range(1, len(tokens)):
                    if tokens[i] == "DLRM_LAST_BOTTOM_LAYER:":
                        self.DLRM_LAST_BOTTOM_LAYER = int(tokens[i+1])
                        if self.generator.id == 0:
                            print(f"****************** info: DLRM workload last bottom layer is: {self.DLRM_LAST_BOTTOM_LAYER}")
            # 处理未识别的并行策略
            elif self.parallelismPolicy == ParallelismPolicy.NONE:
                if not getattr(self, 'PHY_MTP', False):  # 检查类属性PHY_MTP
                    print("######### Exiting because unable to decode the workload parallelization strategy #########", file=sys.stderr)
                    sys.exit(1)
                else:
                    self.parallelismPolicy = ParallelismPolicy.TransformerFwdInBckwd

            # 解析通用维度
            general_involved_dimensions = self.decode_involved_dimensions(
                self.parallelismPolicy, self.model_parallel_npu_group
            )

            # 解析pp_comm_size
            self.pp_commsize = 0
            for i in range(1, len(tokens)):
                if tokens[i] in {"pp_comm", "pp_comm:"}:
                    self.pp_commsize = int(tokens[i+1])
                    break  # 假设只出现一次

            if self.generator.id == 0:
                print(f"pp_commize: {self.pp_commsize}")

            # 参数验证
            if self.generator.id == 0:
                params = [
                    self.model_parallel_npu_group, self.expert_parallel_npu_group,
                    self.pipeline_model_parallelism, self.vpp, self.GA, self.all_gpus
                ]
                pipeline_cond = (self.pipeline_model_parallelism != 1 and self.pp_commsize == 0) or \
                                (self.pipeline_model_parallelism == 1 and self.pp_commsize != 0)
                if any(p == 0 for p in params) or pipeline_cond:
                    print("*****Warining: Input workload format mismatch...*****", file=sys.stderr)

            # 读取运行类型和层数
            self.run_type = tokens[0]
            secondline = inFile.readline().strip()
            lines = int(secondline)
            self.SIZE = lines
            self.layers = []  # 使用列表存储层对象

            for i in range(lines):
                # 读取层基础参数
                parts = inFile.readline().split()
                if not parts:
                    raise ValueError(f"Empty line at layer {i}")
                
                id_ = parts[0]
                depen = int(parts[1])
                fp_compute_time = int(parts[2])
                fp_comm_type_s = parts[3]
                fp_comm_size = int(parts[4])
                ig_compute_time = int(parts[5])
                ig_comm_type_s = parts[6]
                ig_comm_size = int(parts[7])
                wg_compute_time = int(parts[8])
                wg_comm_type_s = parts[9]
                wg_comm_size = int(parts[10])
                wg_update_time = int(parts[11])
                
                specific_policy = ParallelismPolicy.NONE
                selected_involved_dimensions = {}

                wg_type, wg_group_type = ParallelismPolicy.NONE, GroupType.NONE
                ig_type, ig_group_type = ParallelismPolicy.NONE, GroupType.NONE
                fp_type, fp_group_type = ParallelismPolicy.NONE, GroupType.NONE

                if wg_comm_type_s.startswith("ALLREDUCE"):
                    wg_type = ComType.All_Reduce
                    if wg_comm_type_s == "ALLREDUCE":
                        wg_group_type = GroupType.DP
                    elif wg_comm_type_s == "ALLREDUCE_EP":
                        wg_group_type = GroupType.EP
                    elif wg_comm_type_s == "ALLREDUCE_DP_EP":
                        wg_group_type = GroupType.DP_EP
                    else:
                        wg_group_type = GroupType.NONE
                elif wg_comm_type_s.startswith("ALLTOALL"):
                    wg_type = ComType.All_to_All
                    if wg_comm_type_s == "ALLTOALL":
                        wg_group_type = GroupType.DP
                    elif wg_comm_type_s == "ALLTOALL_EP":
                        wg_group_type = GroupType.EP
                    elif wg_comm_type_s == "ALLTOALL_DP_EP":
                        wg_group_type = GroupType.DP_EP
                    else:
                        wg_group_type = GroupType.NONE
                elif wg_comm_type_s.startswith("ALLREDUCEALLTOALL"):
                    wg_type = ComType.All_Reduce_All_to_All
                    if wg_comm_type_s == "ALLREDUCEALLTOALL":
                        wg_group_type = GroupType.DP
                    elif wg_comm_type_s == "ALLREDUCEALLTOALL_EP":
                        wg_group_type = GroupType.EP
                    elif wg_comm_type_s == "ALLREDUCEALLTOALL_DP_EP":
                        wg_group_type = GroupType.DP_EP
                    else:
                        wg_group_type = GroupType.NONE
                elif wg_comm_type_s.startswith("ALLGATHER"):
                    wg_type = ComType.All_Gather
                    if wg_comm_type_s == "ALLGATHER":
                        wg_group_type = GroupType.DP
                    elif wg_comm_type_s == "ALLGATHER_EP":
                        wg_group_type = GroupType.EP
                    elif wg_comm_type_s == "ALLGATHER_DP_EP":
                        wg_group_type = GroupType.DP_EP
                    else:
                        wg_group_type = GroupType.NONE
                elif wg_comm_type_s.startswith("REDUCESCATTER"):
                    wg_type = ComType.Reduce_Scatter
                    if wg_comm_type_s == "REDUCESCATTER":
                        wg_group_type = GroupType.DP
                    elif wg_comm_type_s == "REDUCESCATTER_EP":
                        wg_group_type = GroupType.EP
                    elif wg_comm_type_s == "REDUCESCATTER_DP_EP":
                        wg_group_type = GroupType.DP_EP
                    else:
                        wg_group_type = GroupType.NONE


                if ig_comm_type_s.startswith("ALLREDUCE"):
                    ig_type = ComType.All_Reduce
                    if ig_comm_type_s == "ALLREDUCE":
                        ig_group_type = GroupType.TP
                    elif ig_comm_type_s == "ALLREDUCE_EP":
                        ig_group_type = GroupType.EP
                    elif ig_comm_type_s == "ALLREDUCE_DP_EP":
                        ig_group_type = GroupType.DP_EP
                    else:
                        ig_group_type = GroupType.NONE
                elif ig_comm_type_s.startswith("ALLTOALL"):
                    ig_type = ComType.All_to_All
                    if ig_comm_type_s == "ALLTOALL":
                        ig_group_type = GroupType.TP
                    elif ig_comm_type_s == "ALLTOALL_EP":
                        ig_group_type = GroupType.EP
                    elif ig_comm_type_s == "ALLTOALL_DP_EP":
                        ig_group_type = GroupType.DP_EP
                    else:
                        ig_group_type = GroupType.NONE
                elif ig_comm_type_s.startswith("ALLREDUCEALLTOALL"):
                    ig_type = ComType.All_Reduce_All_to_All
                    if ig_comm_type_s == "ALLREDUCEALLTOALL":
                        ig_group_type = GroupType.TP
                    elif ig_comm_type_s == "ALLREDUCEALLTOALL_EP":
                        ig_group_type = GroupType.EP
                    elif ig_comm_type_s == "ALLREDUCEALLTOALL_DP_EP":
                        ig_group_type = GroupType.DP_EP
                    else:
                        ig_group_type = GroupType.NONE
                elif ig_comm_type_s.startswith("ALLGATHER"):
                    ig_type = ComType.All_Gather
                    if ig_comm_type_s == "ALLGATHER":
                        ig_group_type = GroupType.TP
                    elif ig_comm_type_s == "ALLGATHER_EP":
                        ig_group_type = GroupType.EP
                    elif ig_comm_type_s == "ALLGATHER_DP_EP":
                        ig_group_type = GroupType.DP_EP
                    else:
                        ig_group_type = GroupType.NONE
                elif ig_comm_type_s.startswith("REDUCESCATTER"):
                    ig_type = ComType.Reduce_Scatter
                    if ig_comm_type_s == "REDUCESCATTER":
                        ig_group_type = GroupType.TP
                    elif ig_comm_type_s == "REDUCESCATTER_EP":
                        ig_group_type = GroupType.EP
                    elif ig_comm_type_s == "REDUCESCATTER_DP_EP":
                        ig_group_type = GroupType.DP_EP
                    else:
                        ig_group_type = GroupType.NONE


                if fp_comm_type_s.startswith("ALLREDUCE"):
                    fp_type = ComType.All_Reduce
                    if fp_comm_type_s == "ALLREDUCE":
                        fp_group_type = GroupType.TP
                    elif fp_comm_type_s == "ALLREDUCE_EP":
                        fp_group_type = GroupType.EP
                    elif fp_comm_type_s == "ALLREDUCE_DP_EP":
                        fp_group_type = GroupType.DP_EP
                    else:
                        fp_group_type = GroupType.NONE
                elif fp_comm_type_s.startswith("ALLTOALL"):
                    fp_type = ComType.All_to_All
                    if fp_comm_type_s == "ALLTOALL":
                        fp_group_type = GroupType.TP
                    elif fp_comm_type_s == "ALLTOALL_EP":
                        fp_group_type = GroupType.EP
                    elif fp_comm_type_s == "ALLTOALL_DP_EP":
                        fp_group_type = GroupType.DP_EP
                    else:
                        fp_group_type = GroupType.NONE
                elif fp_comm_type_s.startswith("ALLREDUCEALLTOALL"):
                    fp_type = ComType.All_Reduce_All_to_All
                    if fp_comm_type_s == "ALLREDUCEALLTOALL":
                        fp_group_type = GroupType.TP
                    elif fp_comm_type_s == "ALLREDUCEALLTOALL_EP":
                        fp_group_type = GroupType.EP
                    elif fp_comm_type_s == "ALLREDUCEALLTOALL_DP_EP":
                        fp_group_type = GroupType.DP_EP
                    else:
                        fp_group_type = GroupType.NONE
                elif fp_comm_type_s.startswith("ALLGATHER"):
                    fp_type = ComType.All_Gather
                    if fp_comm_type_s == "ALLGATHER":
                        fp_group_type = GroupType.TP
                    elif fp_comm_type_s == "ALLGATHER_EP":
                        fp_group_type = GroupType.EP
                    elif fp_comm_type_s == "ALLGATHER_DP_EP":
                        fp_group_type = GroupType.DP_EP
                    else:
                        fp_group_type = GroupType.NONE
                elif fp_comm_type_s.startswith("REDUCESCATTER"):
                    fp_type = ComType.Reduce_Scatter
                    if fp_comm_type_s == "REDUCESCATTER":
                        fp_group_type = GroupType.TP
                    elif fp_comm_type_s == "REDUCESCATTER_EP":
                        fp_group_type = GroupType.EP
                    elif fp_comm_type_s == "REDUCESCATTER_DP_EP":
                        fp_group_type = GroupType.DP_EP
                    else:
                        fp_group_type = GroupType.NONE

                if self.generator.id == 0:
                    print(f"id: {id_}, depen: {depen}, wg_comp_time: {wg_compute_time}")

                if self.parallelismPolicy == ParallelismPolicy.HybridCustomized:
                    specific_parallelism = inFile.readline().strip()
                    specific_policy = self.decode_parallelsim(specific_parallelism)

                if (self.parallelismPolicy in {ParallelismPolicy.DLRM, ParallelismPolicy.DLRMEnhanced} and
                    i == 0):
                    specific_policy = ParallelismPolicy.All

                if specific_policy != ParallelismPolicy.NONE:
                    selected_involved_dimensions = self.decode_involved_dimensions(
                        specific_policy, self.model_parallel_npu_group
                    )
                else:
                    selected_involved_dimensions = general_involved_dimensions



                # 创建Layer对象（假设Layer类已实现）
                layer = Layer(
                    id_,
                    i,
                    self.generator,
                    self,
                    fp_compute_time * self.generator.compute_scale,
                    fp_type,
                    fp_group_type,
                    fp_comm_size * self.generator.comm_scale,
                    selected_involved_dimensions["fwd"],
                    ig_compute_time * self.generator.compute_scale,
                    ig_type,
                    ig_group_type,
                    ig_comm_size * self.generator.comm_scale,
                    selected_involved_dimensions["ig"],
                    wg_compute_time * self.generator.compute_scale,
                    wg_type,
                    wg_group_type,
                    wg_comm_size * self.generator.comm_scale,
                    selected_involved_dimensions["wg"],
                    wg_update_time,
                    specific_policy
                )
                # self.layers.append(layer)

                if i in checkpoints:
                    layer.is_checkpoint = True

                if i in need_checkpoint_initiation:
                    layer.needs_fwd_in_bckwd_initiation = True

                self.layers.append(layer)
    
                self.layers[i] = layer

            if self.generator.id == 0:
                print(
                    f"type: {self.run_type}, num passes: {self.TOTAL_PASS}, "
                    f"lines: {lines}, compute scale: {self.generator.compute_scale}, "
                    f"comm scale: {self.generator.comm_scale}"
                )

            return True

        except FileNotFoundError:
            print(f"Unable to open file: {name}", file=sys.stderr)
            print("######### Exiting because unable to open the workload input file #########", file=sys.stderr)
            print("This error is fatal. Please check your path and filename.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading workload file: {str(e)}", file=sys.stderr)
            sys.exit(1)

    def fire(self):
        self.call(EventType.General, None)