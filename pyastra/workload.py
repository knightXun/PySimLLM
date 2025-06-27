import os
import time
import sys
from enum import Enum

import sys
sys.path.append('/data/xla-gpu/PySAI/PySimLLM/pyastra/system')

from system.Common import *
from system.MockNcclGroup import GroupType, ComType
from system.SimData import *
from layer import Layer 
from CsvHelper import CSVWriter 

from ns import ns 

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

class Workload:
    def __init__(self, run_name, name, TOTAL_PASS, total_rows, 
        stat_row, path, seprate_log, total_nodes, flowModel):
        
        self.initialized = False
        self.layers = []
        self.SIZE = 0
        self.run_type = ""
        self.counter = 0
        self.total_nodes = total_nodes
        self.delay_loaded = False
        self.checkpoint_initiated = False
        self.collective_issued = False
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
        
        self.workload_finished_time = 0
        self.workload_compute_time = 0
        self.workload_communicate_time = 0

        self.initialized = self.initialize_workload(name, flowModel)
        if not self.initialized:
            return
        self.total_rows = total_rows
        self.run_name = run_name
        self.registered_for_finished_streams = False

        print(f"stat path: {path} ,total rows: {total_rows} ,stat row: {stat_row}")
        self.detailed = CSVWriter(path, f"detailed_{self.total_nodes}.csv")
        self.end_to_end = CSVWriter(path, "EndToEnd.csv")
        self.dimension_utilization = CSVWriter(path, f"{run_name}_dimension_utilization.csv")
        if stat_row == 0:
            self.initialize_stat_files()

    def initialize_stat_files(self):
        self.detailed.initialize_csv(self.SIZE * self.total_rows + 20, 50)
        self.end_to_end.initialize_csv(self.SIZE * self.total_rows + 20, 50)

    def run(self):
        if self.parallelismPolicy == ParallelismPolicy.DistributedInference:
            self.iterate_distributed_inference()
        elif self.parallelismPolicy == ParallelismPolicy.TransformerFwdInBckwd:
            self.iterate_hybrid_parallel_Transformer_fwd_in_bckwd()
        else:
            raise ValueError("No known parallelism!")


        for i in range(self.SIZE):
            print(f"*************************  {self.layers[i].id}-{i+1} workload stats *************************")
            print(f"layer {i} Total cycles spent on fwd pass compute {self.layers[i].get_fwd_pass_compute()}")
            print(f"layer {i} Total cycles spent on weight grad compute {self.layers[i].get_weight_grad_compute()}")
            print(f"layer {i} Total cycles spent on input grad compute {self.layers[i].get_input_grad_compute()}")
            print(f"layer {i} Total cycles spent on fwd pass comm {self.layers[i].get_fwd_pass_comm()}")
            print(f"layer {i} Total cycles spent on weight grad comm {self.layers[i].get_weight_grad_comm()}")
            print(f"layer {i} Total cycles spent on input grad comm {self.layers[i].get_input_grad_comm()}")

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
        simData = SimData()
        simData.run_name = self.run_name
        # simData.workload_finished_time = Sys.boostedTick() / 1000
        simData.layers_stats = []

        #print(f"workload stats for the job scheduled at NPU offset: {self.generator.npu_offset}")
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
        
            simData.layers_stats.append(layer_stats)

        simData.total_compute = total_compute
        simData.total_exposed_comm = total_exposed
        # simData.avg_chunk_latency_per_logical_dimension = [
        #     latency / Common.FREQ for latency in self.generator.scheduler_unit.get_average_latency_per_dimension()
        # ]
        # print("*************************")

        # if self.seprate_log:
        #     dims = []
        #     for i in range(len(self.generator.scheduler_unit.usage)):
        #         dims.append(self.generator.scheduler_unit.usage[i].report_percentage(10000))
        #     self.dimension_utilization.finalize_csv(dims)
        self.dimension_utilization.finalize_csv()


    def iterate_distributed_inference(self):
        assert self.index >= 0
        assert self.index < self.SIZE

        for i in range(self.SIZE):
            # 先拿到前向计算时间，再加上前向通讯时间
            compute_time = self.layers[i].get_fwd_pass_compute()
            comm_time = self.layers[i].issue_forward_pass_comm()
            self.workload_finished_time += compute_time
            self.workload_finished_time += comm_time

            self.workload_compute_time += compute_time
            self.workload_communicate_time += comm_time

        
        return

    def iterate_hybrid_parallel_Transformer_fwd_in_bckwd(self):
        
        assert self.index >= 0
        assert self.index < self.SIZE

        # 先执行前向
        for i in range(self.SIZE):
            # 先拿到前向计算时间，再加上前向通讯时间
            compute_time = self.layers[i].get_fwd_pass_compute()
            comm_time = self.layers[i].issue_forward_pass_comm()

            self.workload_finished_time += compute_time
            self.workload_finished_time += comm_time

            self.workload_compute_time += compute_time
            self.workload_communicate_time += comm_time
            print(f"Layer {i} FWD compute time: {compute_time}, comm_time: {comm_time}")


        for i in range(self.SIZE-1, -1, -1):
            # 先拿到反向梯度计算时间，再加上反向梯度通讯时间
            compute_time = self.layers[i].get_input_grad_compute()
            comm_time = self.layers[i].issue_input_grad_comm()
            
            self.workload_finished_time += compute_time
            self.workload_finished_time += comm_time

            self.workload_compute_time += compute_time
            self.workload_communicate_time += comm_time

            print(f"Layer {i} Input Grad compute time: {compute_time}, comm_time: {comm_time}")

            # 拿到反向梯度计算时间，再加上反向梯度通讯时间
            compute_time = self.layers[i].get_weight_grad_compute()
            comm_time = self.layers[i].issue_weight_grad_comm()

            self.workload_finished_time += compute_time
            self.workload_finished_time += comm_time
            
            self.workload_compute_time += compute_time
            self.workload_communicate_time += comm_time
            print(f"Layer {i} Weight Grad compute time: {compute_time}, comm_time: {comm_time}")


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
    ):
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
            model_parallel_boundary = 1
            model = [True] * (model_parallel_boundary + 1) + [False] * (9 - model_parallel_boundary)
            data = [False] * (model_parallel_boundary + 1) + [True] * (9 - model_parallel_boundary)
            return {"fwd": model, "ig": model, "wg": data}
        else:
            return {"fwd": none, "ig": none, "wg": none}

    def initialize_workload(self, name: str, flowModel) -> bool:
        
        print(f"Initializing workload from file: {name}")
        checkpoints = {}  
        need_checkpoint_initiation = {}  
        tokens = []
        try:
            with open(name, 'r') as inFile:
                print("Success in opening workload file")

                firstline = inFile.readline().strip()
                tokens = firstline.split()

                # 解析并行策略
                if tokens:
                    self.parallelismPolicy = self.decode_parallelsim(tokens[0])
                else:
                    raise ValueError("First line of workload file is empty")

                # 解析Transformer/TransformerFwdInBckwd策略的参数
                if self.parallelismPolicy in {ParallelismPolicy.Transformer, ParallelismPolicy.TransformerFwdInBckwd}:
                    for i in range(1, len(tokens)):
                        if tokens[i] == "model_parallel_NPU_group:":
                            self.model_parallel_npu_group = int(tokens[i+1])
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
                        print("checkpoints layers are: ")
                        
                        i = 1
                        while i < len(tokens):
                            if tokens[i] == "checkpoints:":
                                count = int(tokens[i+1])
                                while count > 0:
                                    count = count - 1
                                    layer = int(tokens[i+2])
                                    checkpoints[layer] = True
                                    print(f"{layer}, ")
                                
                                i += 1
                            elif tokens[i] == "checkpoint_initiates:":

                                print()
                                print("layers initiating fwd_in_bckwd are: ")
                            
                                count = int(tokens[i+1])
                                while count > 0:
                                    count = count - 1
                                    layer = int(tokens[i+2])
                                    need_checkpoint_initiation[layer] = True
                                    
                                    print(layer, ", ")

                                    print()

                                i += 1
                            else:
                                i += 1
                        if "checkpoint_initiates:" in tokens:
                            print()

                # 解析DLRM/DLRMEnhanced策略的参数
                elif self.parallelismPolicy in {ParallelismPolicy.DLRM, ParallelismPolicy.DLRMEnhanced}:
                    for i in range(1, len(tokens)):
                        if tokens[i] == "DLRM_LAST_BOTTOM_LAYER:":
                            self.DLRM_LAST_BOTTOM_LAYER = int(tokens[i+1])

                            print(f"****************** info: DLRM workload last bottom layer is: {self.DLRM_LAST_BOTTOM_LAYER}")

                # 处理未识别的并行策略, 默认TransformerFwdInBckwd
                elif self.parallelismPolicy == ParallelismPolicy.NONE:
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

                print(f"pp_commize: {self.pp_commsize}")

                # 参数验证
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
                # import pdb; pdb.set_trace()
                secondline = inFile.readline().strip()
                print("second line is: ", secondline)
                lines = int(secondline)
                self.SIZE = lines
                self.layers = []  # 使用列表存储层对象

                contents = inFile.readlines()
                for i in range( len(contents) ):
                     # 读取层基础参数
                    parts = contents[i].split()
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

                    wg_type, wg_group_type = ComType.NONE, GroupType.NONE
                    ig_type, ig_group_type = ComType.NONE, GroupType.NONE
                    fp_type, fp_group_type = ComType.NONE, GroupType.NONE

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

                    print(f"id: {id_}, depen: {depen}, wg_comp_time: {wg_compute_time}")

                    # if self.parallelismPolicy == ParallelismPolicy.HybridCustomized:
                    #     specific_parallelism = inFile.readline().strip()
                    #     specific_policy = self.decode_parallelsim(specific_parallelism)

                    # if (self.parallelismPolicy in {ParallelismPolicy.DLRM, ParallelismPolicy.DLRMEnhanced} and
                    #     i == 0):
                    #     specific_policy = ParallelismPolicy.All

                    # if specific_policy != ParallelismPolicy.NONE:
                    #     selected_involved_dimensions = self.decode_involved_dimensions(
                    #         specific_policy, self.model_parallel_npu_group
                    #     )
                    # else:
                    #     selected_involved_dimensions = general_involved_dimensions


                    self.compute_scale = 1
                    self.comm_scale = 1

                    layer = Layer(
                        id_,
                        i,
                        fp_compute_time * self.compute_scale,
                        fp_type,
                        fp_group_type,
                        fp_comm_size * self.comm_scale,
                        None, # selected_involved_dimensions["fwd"],
                        ig_compute_time * self.compute_scale,
                        ig_type,
                        ig_group_type,
                        ig_comm_size * self.comm_scale,
                        None, # selected_involved_dimensions["ig"],
                        wg_compute_time * self.compute_scale,
                        wg_type,
                        wg_group_type,
                        wg_comm_size * self.comm_scale,
                        None, # selected_involved_dimensions["wg"],
                        wg_update_time,
                        specific_policy,
                        flowModel
                    )
                    # self.layers.append(layer)

                    if i in checkpoints:
                        layer.is_checkpoint = True

                    if i in need_checkpoint_initiation:
                        layer.needs_fwd_in_bckwd_initiation = True

                    self.layers.append(layer)
        
                    self.layers[i] = layer

            print(
                f"type: {self.run_type}, num passes: {self.TOTAL_PASS}, "
                f"lines: {lines}, compute scale: {self.compute_scale}, "
                f"comm scale: {self.comm_scale}"
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

if __name__ == "__main__":

    def test1():
        import time 
        begint = time.time_ns()
        import net
        from net import ReadConf, SetConfig, SetupNetwork
        ReadConf("etc/Spectrum-X_8g_8gps_400Gbps_H100", "etc/SimAI.conf")
        print("Read Conf Done.")
        SetConfig()
        SetupNetwork(None, None) 
        from FlowModel import FlowModel
        nodes = list(range(8))
        NVswitchs = list(range(8,9))

        flowModel = FlowModel(nodes, NVswitchs, net.n, \
            net.portNumber, net.pairBdp, net.has_win, \
            net.global_t, net.pairRtt, net.maxRtt, \
            net.serverAddress, net.maxBdp)

        print("Running Simulation.")
        # allreduceTime = flowModel.runAllReduce(5120)
        
        w = Workload("G175B", "workloads/G175B-M1-C03_GPT175B_megatron_tp8_pp1_mbs1_A100.txt", 1, 1, 0, "etc", False, 8, flowModel)
        # w.initialize_workload("G175B", flowModel)
        w.run()

        print(f"8 nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
        ns.Simulator.Destroy()
        print("Simulation Done.")

    def test2():
        import time 
        begint = time.time_ns()
        import net
        from net import ReadConf, SetConfig, SetupNetwork
        ReadConf("etc/Spectrum-X_16g_8gps_100Gbps_A100", "etc/SimAI.conf")
        print("Read Conf Done.")
        SetConfig()
        SetupNetwork(None, None) 
        from FlowModel import FlowModel
        nodes = list(range(8))
        NVswitchs = list(range(8,9))

        flowModel = FlowModel(nodes, NVswitchs, net.n, \
            net.portNumber, net.pairBdp, net.has_win, \
            net.global_t, net.pairRtt, net.maxRtt, \
            net.serverAddress, net.maxBdp)

        print("Running Simulation.")
        # allreduceTime = flowModel.runAllReduce(5120)
        
        w = Workload("G175B", "workloads/G175B-M1-C03_GPT175B_megatron_tp8_pp1_mbs1_A100.txt", 1, 1, 0, "etc", False, 8, flowModel)
        # w.initialize_workload("G175B", flowModel)
        w.run()

        print(f"16 nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
        ns.Simulator.Destroy()
        print("Simulation Done.")

    def test3():
        import time 
        begint = time.time_ns()
        import net
        from net import ReadConf, SetConfig, SetupNetwork
        ReadConf("etc/Spectrum-X_128g_8gps_100Gbps_A100", "etc/SimAI.conf")
        print("Read Conf Done.")
        SetConfig()
        SetupNetwork(None, None) 
        from FlowModel import FlowModel
        nodes = list(range(8))
        NVswitchs = list(range(8,9))

        flowModel = FlowModel(nodes, NVswitchs, net.n, \
            net.portNumber, net.pairBdp, net.has_win, \
            net.global_t, net.pairRtt, net.maxRtt, \
            net.serverAddress, net.maxBdp)

        print("Running Simulation.")
        # allreduceTime = flowModel.runAllReduce(5120)
        
        w = Workload("G175B", "workloads/G175B-M1-C03_GPT175B_megatron_tp8_pp1_mbs1_A100.txt", 1, 1, 0, "etc", False, 8, flowModel)
        # w.initialize_workload("G175B", flowModel)
        w.run()

        print(f"128 nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
        ns.Simulator.Destroy()
        print("Simulation Done.")

    def test4():
        import time 
        begint = time.time_ns()
        import net
        from net import ReadConf, SetConfig, SetupNetwork
        ReadConf("etc/Spectrum-X_8g_8gps_400Gbps_H100", "etc/SimAI.conf")
        print("Read Conf Done.")
        SetConfig()
        SetupNetwork(None, None) 
        from FlowModel import FlowModel
        nodes = list(range(8))
        NVswitchs = list(range(8,9))

        flowModel = FlowModel(nodes, NVswitchs, net.n, \
            net.portNumber, net.pairBdp, net.has_win, \
            net.global_t, net.pairRtt, net.maxRtt, \
            net.serverAddress, net.maxBdp)

        print("Running Simulation.")
        # allreduceTime = flowModel.runAllReduce(5120)
        
        w = Workload("G13B", "workloads/G13B-M1-C01_GPT13B_megatron_tp8_pp1_mbs1_A100.txt", 1, 1, 0, "etc", False, 8, flowModel)
        # w.initialize_workload("G175B", flowModel)
        w.run()

        print(f"8 nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
        ns.Simulator.Destroy()
        print("Simulation Done.")

    def test5():
        import time 
        begint = time.time_ns()
        import net
        from net import ReadConf, SetConfig, SetupNetwork
        ReadConf("etc/Spectrum-X_16g_8gps_100Gbps_A100", "etc/SimAI.conf")
        print("Read Conf Done.")
        SetConfig()
        SetupNetwork(None, None) 
        from FlowModel import FlowModel
        nodes = list(range(16))
        NVswitchs = list(range(17,19))

        flowModel = FlowModel(nodes, NVswitchs, net.n, \
            net.portNumber, net.pairBdp, net.has_win, \
            net.global_t, net.pairRtt, net.maxRtt, \
            net.serverAddress, net.maxBdp)

        print("Running Simulation.")
        # allreduceTime = flowModel.runAllReduce(5120)
        
        w = Workload("G13B", "workloads/G13B-M1-C01_GPT13B_megatron_tp8_pp1_mbs1_A100.txt", 1, 1, 0, "etc", False, 16, flowModel)
        # w.initialize_workload("G175B", flowModel)
        w.run()

        print(f"16 nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
        ns.Simulator.Destroy()
        print("Simulation Done.")

    def test6():
        import time 
        begint = time.time_ns()
        import net
        from net import ReadConf, SetConfig, SetupNetwork
        ReadConf("etc/Spectrum-X_128g_8gps_100Gbps_A100", "etc/SimAI.conf")
        print("Read Conf Done.")
        SetConfig()
        SetupNetwork(None, None) 
        from FlowModel import FlowModel
        nodes = list(range(128))
        NVswitchs = list(range(129,144))

        flowModel = FlowModel(nodes, NVswitchs, net.n, \
            net.portNumber, net.pairBdp, net.has_win, \
            net.global_t, net.pairRtt, net.maxRtt, \
            net.serverAddress, net.maxBdp)

        print("Running Simulation.")
        
        w = Workload("G13B", "workloads/G13B-M1-C01_GPT13B_megatron_tp8_pp1_mbs1_A100.txt", 1, 1, 0, "etc", False, 128, flowModel)
        w.run()

        print(f"128 nodes simulation total time: {w.workload_finished_time}, compute time: {w.workload_compute_time}, comm time: {w.workload_communicate_time}")
        ns.Simulator.Destroy()
        print("Simulation Done.")

    # print("*" * 20)
    # test4()
    # print("*" * 20)
    # test5()
    # print("*" * 20)
    test6()