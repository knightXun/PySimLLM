import math
import csv
import time
from pathlib import Path
import math

from collections import defaultdict

import sys
sys.path.append('/data/xla-gpu/PySAI/PySimLLM/pyastra')

from system import Common
from system.Common import * 
from system.Common import SchedulingPolicy, CollectiveBarrier
from system.MockNcclGroup import GroupType
from CsvHelper import CSVWriter
from system.SimData import LayerData
# from system.SimData import LayerData

class Layer:
    def __init__(
        self,
        id,
        layer_num,
        fwd_pass_compute_time,
        fwd_pass_comm_type,
        fwd_pass_group_type,
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        input_grad_compute_time,
        input_grad_comm_type,
        input_grad_group_type,
        input_grad_comm_size,
        input_grad_comm_involved_dimensions,
        weight_grad_compute_time,
        weight_grad_comm_type,
        weight_grad_group_type,
        weight_grad_comm_size,
        weight_grad_comm_involved_dimensions,
        weight_grad_update_time,
        specific_policy,
        flowModel
    ):
        self.id = id
        self.layer_num = layer_num
        self.fwd_pass_compute_time = fwd_pass_compute_time
        self.fwd_pass_comm_type = ComType(fwd_pass_comm_type)
        self.fwd_pass_group_type = fwd_pass_group_type
        self.fwd_pass_comm_size = fwd_pass_comm_size
        self.fwd_pass_comm_involved_dimensions = fwd_pass_comm_involved_dimensions
        self.input_grad_compute_time = input_grad_compute_time
        self.input_grad_comm_type = ComType(input_grad_comm_type)
        self.input_grad_group_type = input_grad_group_type
        self.input_grad_comm_size = input_grad_comm_size
        self.input_grad_comm_involved_dimensions = input_grad_comm_involved_dimensions
        self.weight_grad_compute_time = weight_grad_compute_time
        self.weight_grad_comm_type = ComType( weight_grad_comm_type )
        self.weight_grad_group_type = weight_grad_group_type
        self.weight_grad_comm_size = weight_grad_comm_size
        self.weight_grad_comm_involved_dimensions = weight_grad_comm_involved_dimensions
        self.collective_counter = 0
        self.weight_grad_update_time = weight_grad_update_time
        self.weight_grad_update_time = weight_grad_update_time
        self.fwd_update_time = weight_grad_update_time
        self.input_grad_update_time = weight_grad_update_time

        self.total_forward_pass_compute = 0
        self.total_input_grad_compute = 0
        self.total_weight_grad_compute = 0
        self.total_weight_grad_comm = 0
        self.total_input_grad_comm = 0
        self.total_fwd_comm = 0

        self.total_waiting_for_wg_comm = 0
        self.total_waiting_for_ig_comm = 0
        self.total_waiting_for_fwd_comm = 0

        self.last_fwd_finished = 0
        self.last_ig_finished = 0
        self.last_wg_finished = 0
        self.needs_fwd_in_bckwd_initiation = False
        self.is_checkpoint = False
        self.specific_parallellism = specific_policy
        self.lookup_table_size = 0


        self.fwd_pass_datasets = {}
        self.started_waiting_for_fwd_pass = []
        self.input_grad_datasets = {}
        self.started_waiting_for_input_grad = []
        self.weight_grad_datasets = {}
        self.started_waiting_for_weight_grad = []

        self.fwd_barrier = None
        self.wg_barrier = None
        self.ig_barrier = None
        self.flowModel = flowModel

    def get_fwd_pass_compute(self):
        self.total_forward_pass_compute += self.fwd_pass_compute_time
        return self.fwd_pass_compute_time

    def get_input_grad_compute(self):
        self.total_input_grad_compute += self.input_grad_compute_time
        return self.input_grad_compute_time

    def get_weight_grad_compute(self):
        self.total_weight_grad_compute += self.weight_grad_compute_time
        return self.weight_grad_compute_time

    def get_fwd_pass_comm(self):
        return self.fwd_update_time

    def get_input_grad_comm(self):
        return self.input_grad_update_time

    def get_weight_grad_comm(self):
        return self.weight_grad_update_time

    def increment_waiting_for_wg(self):
        self.total_waiting_for_wg_comm += 1

    def increment_waiting_for_ig(self):
        self.total_waiting_for_ig_comm += 1

    def increment_waiting_for_fwd(self):
        self.total_waiting_for_fwd_comm += 1

    def is_fwd_pass_comm_finished(self):
        return len(self.fwd_pass_datasets) == 0

    def is_fwd_pass_comm_finished_blocking(self):
        if len(self.fwd_pass_datasets) == 0:
            return True
        if len(self.started_waiting_for_fwd_pass) == 0:
            self.started_waiting_for_fwd_pass.append(Sys.boostedTick())
        return False

    def is_input_grad_comm_finished(self):
        return len(self.input_grad_datasets) == 0

    def is_input_grad_comm_finished_blocking(self):
        if len(self.input_grad_datasets) == 0:
            return True
        if len(self.started_waiting_for_input_grad) == 0:
            self.started_waiting_for_input_grad.append(Sys.boostedTick())
        return False

    def is_weight_grad_comm_finished(self):
        return len(self.weight_grad_datasets) == 0

    def print_involved_dimensions(self, involved_dimensions):
        print(" involved dimensions: ", end="")
        for dim in involved_dimensions:
            print(" 1," if dim else " 0,", end="")
        print()

    def report(
        self,
        run_name,
        layer_num,
        total_rows,
        stat_row,
        detailed,
        EndToEnd,
        total_compute,
        total_exposed,
        seprate_log,
        total_fwd_time=None,
        total_wg_time=None,
        total_ig_time=None,
        pre_bubble_time=0.0,
        DP_comm=0.0,
        DP_EP_comm=0.0,
        Expose_TP_comm=0.0,
        Expose_EP_comm=0.0
    ):
        if total_fwd_time is None:
            total_fwd_time = [0, 0, 0]
        if total_wg_time is None:
            total_wg_time = [0, 0, 0]
        if total_ig_time is None:
            total_ig_time = [0, 0, 0]

        layerData = LayerData()
        self.take_stream_stats_average()
        TP_size = self.workload.model_parallel_npu_group
        PP_size = self.workload.pipeline_model_parallelism
        DP_size = self.generator.all_gpus[0] // (TP_size * PP_size)
        EP_size = self.workload.expert_parallel_npu_group
        vpp = self.workload.vpp
        pp_commsize = self.workload.pp_commsize
        GA = self.workload.GA
        param = UserParam.getInstance()
        input_grad_group_size = EP_size if self.input_grad_group_type == GroupType.EP else TP_size
        fwd_pass_group_size = EP_size if self.fwd_pass_group_type == GroupType.EP else TP_size
        weight_grad_group_size = DP_size // EP_size if self.weight_grad_group_type == GroupType.DP_EP else DP_size

        if self.id != "embedding_layer":
            pre_bubble_time += ((self.total_waiting_for_fwd_comm + self.total_forward_pass_compute + self.total_weight_grad_compute + self.total_input_grad_compute + self.total_waiting_for_ig_comm) / Common.FREQ)

        if self.weight_grad_group_type == GroupType.DP_EP:
            DP_EP_comm += (self.total_waiting_for_wg_comm / Common.FREQ)
        else:
            DP_comm += (self.total_waiting_for_wg_comm / Common.FREQ)

        if self.fwd_pass_group_type == GroupType.EP:
            Expose_EP_comm += ((self.total_waiting_for_fwd_comm + self.total_waiting_for_ig_comm) / Common.FREQ)
        else:
            Expose_TP_comm += ((self.total_waiting_for_fwd_comm + self.total_waiting_for_ig_comm) / Common.FREQ)

        total_compute += (self.total_forward_pass_compute / Common.FREQ)
        total_compute += (self.total_weight_grad_compute / Common.FREQ)
        total_compute += (self.total_input_grad_compute / Common.FREQ)
        total_exposed += (self.total_waiting_for_fwd_comm / Common.FREQ)
        total_exposed += (self.total_waiting_for_wg_comm / Common.FREQ)
        total_exposed += (self.total_waiting_for_ig_comm / Common.FREQ)

        layerData.layer_name = self.id
        layerData.total_forward_pass_compute = self.total_forward_pass_compute / Common.FREQ
        layerData.total_weight_grad_compute = self.total_weight_grad_compute / Common.FREQ
        layerData.total_input_grad_compute = self.total_input_grad_compute / Common.FREQ
        layerData.total_waiting_for_fwd_comm = self.total_waiting_for_fwd_comm / Common.FREQ
        layerData.total_waiting_for_wg_comm = self.total_waiting_for_wg_comm / Common.FREQ
        layerData.total_waiting_for_ig_comm = self.total_waiting_for_ig_comm / Common.FREQ
        layerData.total_fwd_comm = self.total_fwd_comm / Common.FREQ
        layerData.total_weight_grad_comm = self.total_weight_grad_comm / Common.FREQ
        layerData.total_input_grad_comm = self.total_input_grad_comm / Common.FREQ

        i = 0
        queuing_delay = []
        net_message_latency = []
        for qd in queuing_delay:
            layerData.avg_queuing_delay.append((i, qd / Common.FREQ))
            # i += 1
        
        i = 1
        for ml in net_message_latency:
            layerData.avg_network_message_dealy.append((i, ml / Common.FREQ))
            # i += 1

        if seprate_log:
            data = ""
            total_bw = (0.0, 0.0)
            print("*******************")
            print(f"Layer id: {self.id}")
            print(f"Total collectives issued for this layer: {self.collective_counter}")
            print(f"*************************  Workload stats  ************************* {self.id}")

            if stat_row == 0 and layer_num == 0:
                data = f"layer_name,{run_name},fwd compute,wg compute,ig compute,fwd exposed comm,wg exposed comm,ig exposed comm,fwd total comm,algbw,busbw,wg total comm,algbw,busbw,ig total comm,algbw,busbw"
                EndToEnd.write_line(data)

            data = ""
            if stat_row == 0:
                data += self.id
            data = data + f",{run_name}"

            def format_value(value):
                if math.isfinite(value):
                    return f"{value:.0f}"
                return "NaN or Inf"

            def format_value_bs(value):
                return f"{value:.2f}"

            print(f"id: {self.id} ,Total cycles spent on fwd pass compute: {format_value(self.total_forward_pass_compute / Common.FREQ)}")
            data = data + f",{format_value(self.total_forward_pass_compute / Common.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent on weight grad compute: {format_value(self.total_weight_grad_compute / Common.FREQ)}")
            data = data + f",{format_value(self.total_weight_grad_compute / Common.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent on input grad compute: {format_value(self.total_input_grad_compute / Common.FREQ)}")
            data = data + f",{format_value(self.total_input_grad_compute / Common.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent idle waiting for fwd finish: {format_value(self.total_waiting_for_fwd_comm / Common.FREQ)}")
            data = data + f",{format_value(self.total_waiting_for_fwd_comm / Common.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent idle waiting for weight grad finish: {format_value(self.total_waiting_for_wg_comm / Common.FREQ)}")
            data = data + f",{format_value(self.total_waiting_for_wg_comm / Common.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent idle waiting for input grad finish: {format_value(self.total_waiting_for_ig_comm / Common.FREQ)}")
            data = data + f",{format_value(self.total_waiting_for_ig_comm / Common.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent on fwd pass comm: {format_value(self.total_fwd_comm / Common.FREQ)}")
            total_bw = self.compute_busbw(self.fwd_pass_comm_type, fwd_pass_group_size, self.fwd_pass_comm_size, self.total_fwd_comm)
            data = data + f",{format_value(self.total_fwd_comm / Common.FREQ)}"
            data = data + f",{format_value_bs(total_bw[0])}"
            data = data + f",{format_value_bs(total_bw[1])}"

            print(f"id: {self.id} ,Total cycles spent on weight grad comm: {format_value(self.total_weight_grad_comm / Common.FREQ)}")
            total_bw = self.compute_busbw(self.weight_grad_comm_type, weight_grad_group_size, self.weight_grad_comm_size, self.total_weight_grad_comm)
            data = data + f",{format_value(self.total_weight_grad_comm / Common.FREQ)}"
            data = data + f",{format_value_bs(total_bw[0])}"
            data = data + f",{format_value_bs(total_bw[1])}"

            print(f"id: {self.id} ,Total cycles spent on input grad comm: {format_value(self.total_input_grad_comm / Common.FREQ)}")
            total_bw = self.compute_busbw(self.input_grad_comm_type, input_grad_group_size, self.input_grad_comm_size, self.total_input_grad_comm)
            data = data + f",{format_value(self.total_input_grad_comm / Common.FREQ)}"
            data = data + f",{format_value_bs(total_bw[0])}"
            data = data + f",{format_value_bs(total_bw[1])}"

            EndToEnd.write_line(data)

            def format_value(value):
                if math.isfinite(value):
                    return f"{value:.0f}"
                else:
                    return "NaN or Inf"

            def format_percentage(value, total_time):
                percentage = (value / total_time) * 100
                return f"{percentage:.2f}%"

            # 主逻辑
            data = f"layer_name,{run_name},fwd compute,wg compute,ig compute,fwd exposed comm,wg exposed comm,ig exposed comm,fwd total comm,algbw,busbw,wg total comm,algbw,busbw,ig total comm,algbw,busbw,workload finished at"

            if layer_num == self.workload.SIZE - 1:
                total_exposed = (((float(Sys.boostedTick())) / Common.FREQ) - total_compute)
                data = f"SUM,{run_name},{total_fwd_time[0]},{total_wg_time[0]},{total_ig_time[0]},{total_fwd_time[1]},{total_wg_time[1]},{total_ig_time[1]},{total_fwd_time[2]},NONE,NONE,{total_wg_time[2]},NONE,NONE,{total_ig_time[2]},NONE,NONE"
                EndToEnd.write_line(data)
                
                total_time = total_compute + total_exposed
                data = f"total exposed comm,{total_exposed},total comp,{total_compute},total time,{total_time}"
                EndToEnd.write_line(data)

                # 计算Expose_PP_time（流水线暴露通信时间）
                Expose_PP_time = (2 * vpp * GA * (pp_commsize * Common.GBps / (param.net_work_param.pp) * 1e9) / Common.FREQ)
                Expose_PP_time *= (1 - param.net_work_param.pp_overlap_ratio)
                
                # 计算气泡时间（流水线停顿时间）
                pre_bubble_time *= float(PP_size - 1) / (GA * vpp)
                
                # 构建详细统计数据
                keys = "File name, Expose DP comm, Expose DP_EP comm, Expose TP comm, Expose_EP_comm, Expose_PP_comm, bubble time, total comp, total exposed comm, Total time"
                values = (f"{run_name}, "
                        f"{format_value(DP_comm)} ({format_percentage(DP_comm, total_time)}), "
                        f"{format_value(DP_EP_comm)} ({format_percentage(DP_EP_comm, total_time)}), "
                        f"{format_value(Expose_TP_comm)} ({format_percentage(Expose_TP_comm, total_time)}), "
                        f"{format_value(Expose_EP_comm)} ({format_percentage(Expose_EP_comm, total_time)}), "
                        f"{format_value(Expose_PP_time)} ({format_percentage(Expose_PP_time, total_time)}), "
                        f"{format_value(pre_bubble_time)} ({format_percentage(pre_bubble_time, total_time)}), "
                        f"{format_value(total_compute)} ({format_percentage(total_compute, total_time)}), "
                        f"{format_value(total_exposed)} ({format_percentage(total_exposed, total_time)}), "
                        f"{format_value(total_time)}")
                
                data = f"{keys}\n{values}"
                EndToEnd.write_res(data)

        return layerData

    def report(
        self,
        run_name: str,
        layer_num: int,
        total_rows: int,
        stat_row: int,
        detailed: CSVWriter,
        EndToEnd: CSVWriter,
        total_compute: float,
        total_exposed: float,
        pre_bubble_time: float,
        DP_comm: float,
        DP_EP_comm: float,
        Expose_TP_comm: float,
        Expose_EP_comm: float,
        seprate_log: bool
    ) -> LayerData:
        layerData = LayerData()
        self.take_stream_stats_average()

        TP_size = self.workload.model_parallel_npu_group
        PP_size = self.workload.pipeline_model_parallelism
        vpp = self.workload.vpp
        pp_commsize = self.workload.pp_commsize
        DP_size = self.generator.all_gpus[0] // (TP_size * PP_size)
        GA = self.workload.GA
        EP_size = self.workload.expert_parallel_npu_group
        param = self.param

        # 计算分组大小
        input_grad_group_size = EP_size if self.input_grad_group_type == GroupType.EP else TP_size
        fwd_pass_group_size = EP_size if self.fwd_pass_group_type == GroupType.EP else TP_size
        weight_grad_group_size = None
        if self.weight_grad_group_type == GroupType.DP_EP:
            weight_grad_group_size = DP_size // EP_size
        else:
            weight_grad_group_size = DP_size


        total_waiting_for_fwd_comm = 0.0
        total_waiting_for_ig_comm = 0.0
        total_waiting_for_wg_comm = 0.0

        # 计算气泡时间（非embedding层）
        if self.id != "embedding_layer":
            pre_bubble_time += (
                total_waiting_for_fwd_comm + self.total_forward_pass_compute +
                self.total_weight_grad_compute + self.total_input_grad_compute +
                total_waiting_for_ig_comm
            ) / Common.FREQ

        # 处理权重梯度通信类型
        if self.weight_grad_group_type == GroupType.DP_EP:
            total_waiting_for_wg_comm *= (1 - param.net_work_param.dp_overlap_ratio)
            DP_EP_comm += total_waiting_for_wg_comm / Common.FREQ
        else:
            total_waiting_for_wg_comm *= (1 - param.net_work_param.dp_overlap_ratio)
            DP_comm += total_waiting_for_wg_comm / Common.FREQ

        # 处理前向传播分组类型
        if self.fwd_pass_group_type == GroupType.EP:
            total_waiting_for_fwd_comm *= (1 - param.net_work_param.ep_overlap_ratio)
            total_waiting_for_ig_comm *= (1 - param.net_work_param.ep_overlap_ratio)
            Expose_EP_comm += (total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / Common.FREQ
        else:
            total_waiting_for_fwd_comm *= (1 - param.net_work_param.tp_overlap_ratio)
            total_waiting_for_ig_comm *= (1 - param.net_work_param.tp_overlap_ratio)
            Expose_TP_comm += (total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / Common.FREQ

        # 累计计算和暴露时间
        total_compute += (
            self.total_forward_pass_compute + self.total_weight_grad_compute +
            self.total_input_grad_compute
        ) / Common.FREQ
        total_exposed += (
            total_waiting_for_fwd_comm + total_waiting_for_wg_comm + total_waiting_for_ig_comm
        ) / Common.FREQ

        # 填充LayerData
        layerData.layer_name = self.id
        layerData.total_forward_pass_compute = self.total_forward_pass_compute / Common.FREQ
        layerData.total_weight_grad_compute = self.total_weight_grad_compute / Common.FREQ
        layerData.total_input_grad_compute = self.total_input_grad_compute / Common.FREQ
        layerData.total_waiting_for_fwd_comm = total_waiting_for_fwd_comm / Common.FREQ
        layerData.total_waiting_for_wg_comm = total_waiting_for_wg_comm / Common.FREQ
        layerData.total_waiting_for_ig_comm = total_waiting_for_ig_comm / Common.FREQ
        layerData.total_fwd_comm = self.total_fwd_comm / Common.FREQ
        layerData.total_weight_grad_comm = self.total_weight_grad_comm / Common.FREQ
        layerData.total_input_grad_comm = self.total_input_grad_comm / Common.FREQ

        # 填充排队延迟和网络消息延迟
        layerData.avg_queuing_delay = [(i, qd / Common.FREQ) for i, qd in enumerate(self.queuing_delay)]
        layerData.avg_network_message_dealy = [(i, ml / Common.FREQ) for i, ml in enumerate(self.net_message_latency, start=1)]

        # 处理日志输出
        if seprate_log:
            data = []
            if stat_row == 0 and layer_num == 0:
                header = f"layer_name,{run_name},fwd compute,wg compute,ig compute,fwd exposed comm,wg exposed comm,ig exposed comm,fwd total comm,algbw,busbw,wg total comm,algbw,busbw,ig total comm,algbw,busbw"
                EndToEnd.write_line(header)

            # 格式化数值函数
            def format_value(value: float) -> str:
                return f"{value:.0f}" if math.isfinite(value) else "NaN or Inf"

            def format_value_bs(value: float) -> str:
                return f"{value:.2f}"

            total_bw1 = self.compute_busbw(self.fwd_pass_comm_type, fwd_pass_group_size, 
                self.fwd_pass_comm_size, self.total_fwd_comm);

            total_bw2 = self.compute_busbw(self.weight_grad_comm_type, weight_grad_group_size, 
                self.weight_grad_comm_size, self.total_weight_grad_comm);

            total_bw3 = self.compute_busbw(self.input_grad_comm_type, 
                input_grad_group_size, self.input_grad_comm_size, self.total_input_grad_comm);

            # 构建数据行
            row = [
                self.id if stat_row == 0 else "",
                run_name,
                format_value(self.total_forward_pass_compute / Common.FREQ),
                format_value(self.total_weight_grad_compute / Common.FREQ),
                format_value(self.total_input_grad_compute / Common.FREQ),
                format_value(total_waiting_for_fwd_comm / Common.FREQ),
                format_value(total_waiting_for_wg_comm / Common.FREQ),
                format_value(total_waiting_for_ig_comm / Common.FREQ),
                format_value(self.total_fwd_comm / Common.FREQ),
                format_value_bs(total_bw1[0]), 
                format_value_bs(total_bw1[1]), 
                format_value(self.total_weight_grad_comm / Common.FREQ),
                format_value_bs(total_bw2[0]), 
                format_value_bs(total_bw2[1]),  
                format_value(self.total_input_grad_comm / Common.FREQ),
                format_value_bs(total_bw3[0]),
                format_value_bs(total_bw3[1]), 
            ]
            data_line = ",".join(row)
            EndToEnd.write_line(data_line)

            # 处理最后一层
            if layer_num == self.workload.SIZE - 1:
                if param.mode != ModeType.ANALYTICAL:
                    total_exposed = (self.Sys_boostedTick() / Common.FREQ) - total_compute  # 需实现Sys_boostedTick()

                # 计算PP相关时间
                Expose_PP_time = (
                    2 * vpp * GA * (pp_commsize * Common.GBps / (param.net_work_param.pp) * 1e9) / Common.FREQ
                ) * (1 - param.net_work_param.pp_overlap_ratio)
                pre_bubble_time *= float(PP_size - 1) / (GA * vpp)
                total_time = total_compute + total_exposed + pre_bubble_time + Expose_PP_time

                # 格式化百分比函数
                def format_percentage(value: float) -> str:
                    return f"{(value / total_time) * 100:.2f}%"

                # 提取文件名
                file_name = Path(param.res).name

                # 构建统计摘要
                keys = "File name, Expose DP comm, Expose DP_EP comm, Expose TP comm, Expose_EP_comm, Expose_PP_comm, bubble time, total comp, total exposed comm, Total time"
                values = (
                    f"{file_name}, "
                    f"{format_value(DP_comm)} ({format_percentage(DP_comm)}), "
                    f"{format_value(DP_EP_comm)} ({format_percentage(DP_EP_comm)}), "
                    f"{format_value(Expose_TP_comm)} ({format_percentage(Expose_TP_comm)}), "
                    f"{format_value(Expose_EP_comm)} ({format_percentage(Expose_EP_comm)}), "
                    f"{format_value(Expose_PP_time)} ({format_percentage(Expose_PP_time)}), "
                    f"{format_value(pre_bubble_time)} ({format_percentage(pre_bubble_time)}), "
                    f"{format_value(total_compute)} ({format_percentage(total_compute)}), "
                    f"{format_value(total_exposed)} ({format_percentage(total_exposed)}), "
                    f"{format_value(total_time)}"
                )
                EndToEnd.write_res(f"{keys}\n{values}")

                # 生成可视化图表（如果需要）
                if param.net_work_param.visual:
                    chart_path = EndToEnd.path  # 假设EndToEnd有path属性
                    html_file = Path(chart_path) / "chart.html"
                    with open(html_file, "w") as f:
                        f.write("""<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh; margin: 0; padding-top: 10%; }
        canvas { width: 50%; max-width: 400px; height: auto; }
        h2 { margin: 5px 0; }
    </style>
</head>
<body>
    <canvas id="myPieChart"></canvas>
    <h2>Total Time: {total_time} ns</h2>
    <h2>Model: {file_name}</h2>
    <script>
        var ctx = document.getElementById('myPieChart').getContext('2d');
        var myPieChart = new Chart(ctx, {{
            type: 'pie',
            data: {{
                labels: ['Expose DP comm', 'Expose DP_EP comm', 'Expose TP comm', 'Expose_EP_comm', 'Total compute', 'PP Bubble time', 'Expose PP comm'],
                datasets: [{{
                    data: [{DP_comm}, {DP_EP_comm}, {Expose_TP_comm}, {Expose_EP_comm}, {total_compute}, {pre_bubble_time}, {Expose_PP_time}],
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#FF5733'],
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                var label = context.label || '';
                                if (label) label += ': ';
                                if (context.parsed !== null) label += context.parsed + ' ns';
                                return label;
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
""".format(
                            total_time=total_time,
                            file_name=file_name,
                            DP_comm=DP_comm,
                            DP_EP_comm=DP_EP_comm,
                            Expose_TP_comm=Expose_TP_comm,
                            Expose_EP_comm=Expose_EP_comm,
                            total_compute=total_compute,
                            pre_bubble_time=pre_bubble_time,
                            Expose_PP_time=Expose_PP_time
                        )
                    )
            print("HTML file created")

        return layerData

    def getFileName(path):
        return Path(path).name
    
    def binarySearch(self):
        pass 

    def take_stream_stats_average(self):
        pass

    def update_stream_stats(self, dataset):
        pass

    def compute_busbw(
        self,
        comtype: ComType,
        nranks: int,
        data_size: int,
        total_comm: float
    ):
        """计算算法带宽和总线带宽"""
        algbw = (data_size / (total_comm / Common.FREQ)) * 1000000 * Common.GBps
        busbw = 0.0
        
        if comtype == ComType.All_Reduce:
            busbw = algbw * 2 * (nranks - 1) / nranks
        elif comtype in (ComType.All_Gather, ComType.Reduce_Scatter, ComType.All_to_All):
            busbw = algbw * (nranks - 1) / nranks
        
        return (algbw, busbw)

    def issue_forward_pass_comm(self):
        fwd_pass_comm_type = self.fwd_pass_comm_type
        fwd_pass_comm_size = self.fwd_pass_comm_size
        
        # 处理不同通信类型
        if fwd_pass_comm_type == ComType.All_Reduce:
            self.fwd_update_time = self.flowModel.runAllReduce(fwd_pass_comm_size)
            return self.fwd_update_time

        elif fwd_pass_comm_type == ComType.All_to_All:
            self.fwd_update_time = self.flowModel.runAll2All(fwd_pass_comm_size)
            return self.fwd_update_time
        elif fwd_pass_comm_type == ComType.All_Gather:
            self.fwd_update_time =  self.flowModel.runAllGather(fwd_pass_comm_size)
            return self.fwd_update_time
        elif fwd_pass_comm_type == ComType.Reduce_Scatter:
            self.fwd_update_time =  self.flowModel.runReduceScatter(fwd_pass_comm_size)
            return self.fwd_update_time
        elif fwd_pass_comm_type == ComType.NONE:
            self.fwd_update_time = 0
            return self.fwd_update_time
        else:
            raise RuntimeError("Unknown collective operation!", fwd_pass_comm_type)


    def issue_input_grad_comm(self):
        """发起输入梯度通信过程"""
        input_grad_comm_type = self.input_grad_comm_type
        input_grad_comm_size = self.input_grad_comm_size

        # 处理不同通信类型
        if input_grad_comm_type == ComType.All_Reduce:
            self.input_grad_update_time =  self.flowModel.runAllReduce(input_grad_comm_size)
            return self.input_grad_update_time
        elif input_grad_comm_type == ComType.All_to_All:
            self.input_grad_update_time =   self.flowModel.runAll2All(input_grad_comm_size)
            return self.input_grad_update_time
        elif input_grad_comm_type == ComType.All_Gather:
            self.input_grad_update_time =   self.flowModel.runAllGather(input_grad_comm_size)
            return self.input_grad_update_time
        elif input_grad_comm_type == ComType.Reduce_Scatter:
            self.input_grad_update_time =  self.flowModel.runReduceScatter(input_grad_comm_size)
            return self.input_grad_update_time
        elif input_grad_comm_type == ComType.NONE:
            self.input_grad_update_time = 0
            return self.input_grad_update_time
        else:
            raise RuntimeError("Unknown collective operation!")      


    def issue_weight_grad_comm(self):
        """发起权重梯度通信过程"""
        comm_type = self.weight_grad_comm_type
        comm_size = self.weight_grad_comm_size

        # print(f"*************************  {self.id} weight_grad_comm *************************")
        # import pdb; pdb.set_trace()
        # 处理不同通信类型
        if comm_type == ComType.All_Reduce:
            self.weight_grad_update_time = self.flowModel.runAllReduce(comm_size)
            return self.weight_grad_update_time
        elif comm_type == ComType.All_to_All:
            self.weight_grad_update_time = self.flowModel.runAll2All(comm_size)
            return self.weight_grad_update_time
        elif comm_type == ComType.All_Gather:
            self.weight_grad_update_time =  self.flowModel.runAllGather(comm_size)
            return self.weight_grad_update_time
        elif comm_type == ComType.Reduce_Scatter:
            # simai 在这里没有做通讯，不知原因
            # self.weight_grad_update_time =  self.flowModel.runReduceScatter(comm_size)
            self.weight_grad_update_time = 0
            return self.weight_grad_update_time
        elif comm_type == ComType.NONE:
            self.weight_grad_update_time = 0
            return self.weight_grad_update_time
        else:
            raise RuntimeError("Unknown collective operation!")      