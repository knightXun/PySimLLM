import math
import csv
import time
from collections import defaultdict

# 假设的类型定义
class EventType:
    Wight_Grad_Comm_Finished = 1
    Input_Grad_Comm_Finished = 2
    Fwd_Comm_Finished = 3
    Wight_Grad_Comm_Finished_After_Delay = 4
    Input_Grad_Comm_Finished_After_Delay = 5
    Fwd_Comm_Finished_After_Delay = 6
    General = 7

class ComType:
    All_Reduce = 1
    All_to_All = 2
    All_Gather = 3
    Reduce_Scatter = 4
    None = 0

class MockNccl:
    class GroupType:
        TP = 1
        EP = 2
        DP = 3
        DP_EP = 4

class CollectiveBarrier:
    Blocking = 1

class SchedulingPolicy:
    pass

class DataSet:
    def __init__(self, my_id, creation_tick, total_streams):
        self.my_id = my_id
        self.creation_tick = creation_tick
        self.finish_tick = creation_tick
        self.total_streams = total_streams
        self.active = True
        self.notifier = None
        self.notify_event = None

    def set_notifier(self, notifier, event):
        self.notifier = notifier
        self.notify_event = event

class IntData:
    def __init__(self, data):
        self.data = data

class CSVWriter:
    def __init__(self, path):
        self.path = path

    def write_line(self, data):
        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.split(','))

    def write_res(self, data):
        with open(self.path, 'a') as file:
            file.write(data + '\n')

class Workload:
    def __init__(self, model_parallel_npu_group, pipeline_model_parallelism, all_gpus, expert_parallel_npu_group, vpp, pp_commsize, GA, SIZE):
        self.model_parallel_npu_group = model_parallel_npu_group
        self.pipeline_model_parallelism = pipeline_model_parallelism
        self.all_gpus = all_gpus
        self.expert_parallel_npu_group = expert_parallel_npu_group
        self.vpp = vpp
        self.pp_commsize = pp_commsize
        self.GA = GA
        self.SIZE = SIZE

    def call(self, event, mdata):
        pass

class Sys:
    id = 0
    FREQ = 1e9
    @staticmethod
    def boostedTick():
        return time.time_ns()

    @staticmethod
    def sys_panic(message):
        raise ValueError(message)

    @staticmethod
    def increase_finished_streams(streams):
        pass

class MockNcclLog:
    instance = None

    @staticmethod
    def getInstance():
        if MockNcclLog.instance is None:
            MockNcclLog.instance = MockNcclLog()
        return MockNcclLog.instance

    def writeLog(self, level, message, *args):
        print(message % args)

class UserParam:
    instance = None

    @staticmethod
    def getInstance():
        if UserParam.instance is None:
            UserParam.instance = UserParam()
        return UserParam.instance

    def __init__(self):
        self.mode = None
        self.net_work_param = type('NetWorkParam', (), {
            'gpus_per_server': 0,
            'gpu_type': None,
            'tp_ar': 0,
            'tp_ag': 0,
            'tp_ata': 0,
            'ep_ata': 0,
            'dp_ag': 0,
            'ep_ag': 0,
            'dp_ar': 0,
            'ep_ar': 0,
            'dp_overlap_ratio': 0,
            'ep_overlap_ratio': 0,
            'tp_overlap_ratio': 0,
            'pp': 0,
            'pp_overlap_ratio': 0,
            'visual': False
        })()
        self.res = ""

class LayerData:
    def __init__(self):
        self.layer_name = ""
        self.total_forward_pass_compute = 0
        self.total_weight_grad_compute = 0
        self.total_input_grad_compute = 0
        self.total_waiting_for_fwd_comm = 0
        self.total_waiting_for_wg_comm = 0
        self.total_waiting_for_ig_comm = 0
        self.total_fwd_comm = 0
        self.total_weight_grad_comm = 0
        self.total_input_grad_comm = 0
        self.avg_queuing_delay = []
        self.avg_network_message_dealy = []

class Layer:
    def __init__(
        self,
        id,
        layer_num,
        generator,
        workload,
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
        specific_policy
    ):
        self.id = id
        self.layer_num = layer_num
        self.generator = generator
        self.workload = workload
        self.fwd_pass_compute_time = fwd_pass_compute_time
        self.fwd_pass_comm_type = fwd_pass_comm_type
        self.fwd_pass_group_type = fwd_pass_group_type
        self.fwd_pass_comm_size = fwd_pass_comm_size
        self.fwd_pass_comm_involved_dimensions = fwd_pass_comm_involved_dimensions
        self.input_grad_compute_time = input_grad_compute_time
        self.input_grad_comm_type = input_grad_comm_type
        self.input_grad_group_type = input_grad_group_type
        self.input_grad_comm_size = input_grad_comm_size
        self.input_grad_comm_involved_dimensions = input_grad_comm_involved_dimensions
        self.weight_grad_compute_time = weight_grad_compute_time
        self.weight_grad_comm_type = weight_grad_comm_type
        self.weight_grad_group_type = weight_grad_group_type
        self.weight_grad_comm_size = weight_grad_comm_size
        self.weight_grad_comm_involved_dimensions = weight_grad_comm_involved_dimensions
        self.collective_counter = 0
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
        assert generator is not None

        self.fwd_pass_datasets = {}
        self.started_waiting_for_fwd_pass = []
        self.input_grad_datasets = {}
        self.started_waiting_for_input_grad = []
        self.weight_grad_datasets = {}
        self.started_waiting_for_weight_grad = []

        self.fwd_barrier = None
        self.wg_barrier = None
        self.ig_barrier = None

    def call(self, event, mdata):
        if event == EventType.Wight_Grad_Comm_Finished:
            self.last_wg_finished = Sys.boostedTick()
            self.generator.register_event(
                self,
                EventType.Wight_Grad_Comm_Finished_After_Delay,
                mdata,
                self.weight_grad_update_time
            )
            return
        elif event == EventType.Input_Grad_Comm_Finished:
            self.last_ig_finished = Sys.boostedTick()
            self.generator.register_event(
                self,
                EventType.Input_Grad_Comm_Finished_After_Delay,
                mdata,
                self.input_grad_update_time
            )
            return
        elif event == EventType.Fwd_Comm_Finished:
            self.last_fwd_finished = Sys.boostedTick()
            self.generator.register_event(
                self,
                EventType.Fwd_Comm_Finished_After_Delay,
                mdata,
                self.fwd_update_time
            )
            return

        data = mdata.data
        if event == EventType.Wight_Grad_Comm_Finished_After_Delay:
            if Sys.id == 0:
                print(f"***** info: weight gradient collective for layer: {self.id} is finished************")
            self.weight_grad_datasets[data].finish_tick += self.weight_grad_update_time
            self.total_weight_grad_comm += self.weight_grad_datasets[data].finish_tick - self.weight_grad_datasets[data].creation_tick

            if len(self.weight_grad_datasets) == 1 and self.wg_barrier == CollectiveBarrier.Blocking:
                self.total_waiting_for_wg_comm += self.weight_grad_datasets[data].finish_tick - self.weight_grad_datasets[data].creation_tick
                self.update_stream_stats(self.weight_grad_datasets[data])
                dataset_streams = self.weight_grad_datasets[data].total_streams
                del self.weight_grad_datasets[data]
                self.workload.call(EventType.General, None)
                Sys.increase_finished_streams(dataset_streams)
                return
            elif len(self.started_waiting_for_weight_grad) > 0:
                self.total_waiting_for_wg_comm += self.weight_grad_datasets[data].finish_tick - self.started_waiting_for_weight_grad[0]
                self.started_waiting_for_weight_grad.pop(0)
                self.update_stream_stats(self.weight_grad_datasets[data])
                dataset_streams = self.weight_grad_datasets[data].total_streams
                del self.weight_grad_datasets[data]
                self.workload.call(EventType.General, None)
                Sys.increase_finished_streams(dataset_streams)
                return

            self.update_stream_stats(self.weight_grad_datasets[data])
            dataset_streams = self.weight_grad_datasets[data].total_streams
            del self.weight_grad_datasets[data]
            Sys.increase_finished_streams(dataset_streams)
        elif event == EventType.Input_Grad_Comm_Finished_After_Delay:
            if Sys.id == 0:
                print(f"***** info: input gradient collective for layer: {self.id} is finished************")
            self.input_grad_datasets[data].finish_tick += self.input_grad_update_time
            self.total_input_grad_comm += self.input_grad_datasets[data].finish_tick - self.input_grad_datasets[data].creation_tick

            if len(self.input_grad_datasets) == 1 and self.ig_barrier == CollectiveBarrier.Blocking:
                self.total_waiting_for_ig_comm += self.input_grad_datasets[data].finish_tick - self.input_grad_datasets[data].creation_tick
                self.update_stream_stats(self.input_grad_datasets[data])
                dataset_streams = self.input_grad_datasets[data].total_streams
                del self.input_grad_datasets[data]
                self.workload.call(EventType.General, None)
                Sys.increase_finished_streams(dataset_streams)
                return
            elif len(self.started_waiting_for_input_grad) > 0:
                self.total_waiting_for_ig_comm += self.input_grad_datasets[data].finish_tick - self.started_waiting_for_input_grad[0]
                self.started_waiting_for_input_grad.pop(0)
                self.update_stream_stats(self.input_grad_datasets[data])
                dataset_streams = self.input_grad_datasets[data].total_streams
                del self.input_grad_datasets[data]
                self.workload.call(EventType.General, None)
                Sys.increase_finished_streams(dataset_streams)
                return

            self.update_stream_stats(self.input_grad_datasets[data])
            dataset_streams = self.input_grad_datasets[data].total_streams
            del self.input_grad_datasets[data]
            Sys.increase_finished_streams(dataset_streams)
        elif event == EventType.Fwd_Comm_Finished_After_Delay:
            if Sys.id == 0:
                print(f"***** info: fwd pass comm collective for layer: {self.id} is finished************")
            self.fwd_pass_datasets[data].finish_tick += self.fwd_update_time
            self.total_fwd_comm += self.fwd_pass_datasets[data].finish_tick - self.fwd_pass_datasets[data].creation_tick

            if len(self.fwd_pass_datasets) == 1 and self.fwd_barrier == CollectiveBarrier.Blocking:
                self.total_waiting_for_fwd_comm += self.fwd_pass_datasets[data].finish_tick - self.fwd_pass_datasets[data].creation_tick
                self.update_stream_stats(self.fwd_pass_datasets[data])
                dataset_streams = self.fwd_pass_datasets[data].total_streams
                del self.fwd_pass_datasets[data]
                self.workload.call(EventType.General, None)
                Sys.increase_finished_streams(dataset_streams)
                return
            elif len(self.started_waiting_for_fwd_pass) > 0:
                self.total_waiting_for_fwd_comm += self.fwd_pass_datasets[data].finish_tick - self.started_waiting_for_fwd_pass[0]
                self.started_waiting_for_fwd_pass.pop(0)
                self.update_stream_stats(self.fwd_pass_datasets[data])
                dataset_streams = self.fwd_pass_datasets[data].total_streams
                del self.fwd_pass_datasets[data]
                self.workload.call(EventType.General, None)
                Sys.increase_finished_streams(dataset_streams)
                return

            self.update_stream_stats(self.fwd_pass_datasets[data])
            dataset_streams = self.fwd_pass_datasets[data].total_streams
            del self.fwd_pass_datasets[data]
            Sys.increase_finished_streams(dataset_streams)

    def get_fwd_pass_compute(self):
        self.total_forward_pass_compute += self.fwd_pass_compute_time
        return self.fwd_pass_compute_time

    def get_input_grad_compute(self):
        self.total_input_grad_compute += self.input_grad_compute_time
        return self.input_grad_compute_time

    def get_weight_grad_compute(self):
        self.total_weight_grad_compute += self.weight_grad_compute_time
        return self.weight_grad_compute_time

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

    def is_weight_grad_comm_finished_blocking(self):
        if len(self.weight_grad_datasets) == 0:
            return True
        if len(self.started_waiting_for_weight_grad) == 0:
            self.started_waiting_for_weight_grad.append(Sys.boostedTick())
        return False

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
        pre_bubble_time=0,
        DP_comm=0,
        DP_EP_comm=0,
        Expose_TP_comm=0,
        Expose_EP_comm=0
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
        input_grad_group_size = EP_size if self.input_grad_group_type == MockNccl.GroupType.EP else TP_size
        fwd_pass_group_size = EP_size if self.fwd_pass_group_type == MockNccl.GroupType.EP else TP_size
        weight_grad_group_size = DP_size // EP_size if self.weight_grad_group_type == MockNccl.GroupType.DP_EP else DP_size

        if self.id != "embedding_layer":
            pre_bubble_time += ((self.total_waiting_for_fwd_comm + self.total_forward_pass_compute + self.total_weight_grad_compute + self.total_input_grad_compute + self.total_waiting_for_ig_comm) / Sys.FREQ)

        if self.weight_grad_group_type == MockNccl.GroupType.DP_EP:
            DP_EP_comm += (self.total_waiting_for_wg_comm / Sys.FREQ)
        else:
            DP_comm += (self.total_waiting_for_wg_comm / Sys.FREQ)

        if self.fwd_pass_group_type == MockNccl.GroupType.EP:
            Expose_EP_comm += ((self.total_waiting_for_fwd_comm + self.total_waiting_for_ig_comm) / Sys.FREQ)
        else:
            Expose_TP_comm += ((self.total_waiting_for_fwd_comm + self.total_waiting_for_ig_comm) / Sys.FREQ)

        total_compute += (self.total_forward_pass_compute / Sys.FREQ)
        total_compute += (self.total_weight_grad_compute / Sys.FREQ)
        total_compute += (self.total_input_grad_compute / Sys.FREQ)
        total_exposed += (self.total_waiting_for_fwd_comm / Sys.FREQ)
        total_exposed += (self.total_waiting_for_wg_comm / Sys.FREQ)
        total_exposed += (self.total_waiting_for_ig_comm / Sys.FREQ)

        layerData.layer_name = self.id
        layerData.total_forward_pass_compute = self.total_forward_pass_compute / Sys.FREQ
        layerData.total_weight_grad_compute = self.total_weight_grad_compute / Sys.FREQ
        layerData.total_input_grad_compute = self.total_input_grad_compute / Sys.FREQ
        layerData.total_waiting_for_fwd_comm = self.total_waiting_for_fwd_comm / Sys.FREQ
        layerData.total_waiting_for_wg_comm = self.total_waiting_for_wg_comm / Sys.FREQ
        layerData.total_waiting_for_ig_comm = self.total_waiting_for_ig_comm / Sys.FREQ
        layerData.total_fwd_comm = self.total_fwd_comm / Sys.FREQ
        layerData.total_weight_grad_comm = self.total_weight_grad_comm / Sys.FREQ
        layerData.total_input_grad_comm = self.total_input_grad_comm / Sys.FREQ

        i = 0
        queuing_delay = []
        net_message_latency = []
        for qd in queuing_delay:
            layerData.avg_queuing_delay.append((i, qd / Sys.FREQ))
            i += 1
        i = 1
        for ml in net_message_latency:
            layerData.avg_network_message_dealy.append((i, ml / Sys.FREQ))
            i += 1

        if seprate_log:
            data = ""
            total_bw = (0, 0)
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

            print(f"id: {self.id} ,Total cycles spent on fwd pass compute: {format_value(self.total_forward_pass_compute / Sys.FREQ)}")
            data = data + f",{format_value(self.total_forward_pass_compute / Sys.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent on weight grad compute: {format_value(self.total_weight_grad_compute / Sys.FREQ)}")
            data = data + f",{format_value(self.total_weight_grad_compute / Sys.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent on input grad compute: {format_value(self.total_input_grad_compute / Sys.FREQ)}")
            data = data + f",{format_value(self.total_input_grad_compute / Sys.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent idle waiting for fwd finish: {format_value(self.total_waiting_for_fwd_comm / Sys.FREQ)}")
            data = data + f",{format_value(self.total_waiting_for_fwd_comm / Sys.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent idle waiting for weight grad finish: {format_value(self.total_waiting_for_wg_comm / Sys.FREQ)}")
            data = data + f",{format_value(self.total_waiting_for_wg_comm / Sys.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent idle waiting for input grad finish: {format_value(self.total_waiting_for_ig_comm / Sys.FREQ)}")
            data = data + f",{format_value(self.total_waiting_for_ig_comm / Sys.FREQ)}"

            print(f"id: {self.id} ,Total cycles spent on fwd pass comm: {format_value(self.total_fwd_comm / Sys.FREQ)}")
            total_bw = self.compute_busbw(self.fwd_pass_comm_type, fwd_pass_group_size, self.fwd_pass_comm_size, self.total_fwd_comm)
            data = data + f",{format_value(self.total_fwd_comm / Sys.FREQ)}"
            data = data + f",{format_value_bs(total_bw[0])}"
            data = data + f",{format_value_bs(total_bw[1])}"

            print(f"id: {self.id} ,Total cycles spent on weight grad comm: {format_value(self.total_weight_grad_comm / Sys.FREQ)}")
            total_bw = self.compute_busbw(self.weight_grad_comm_type, weight_grad_group_size, self.weight_grad_comm_size, self.total_weight_grad_comm)
            data = data + f",{format_value(self.total_weight_grad_comm / Sys.FREQ)}"
            data = data + f",{format_value_bs(total_bw[0])}"
            data = data + f",{format_value_bs(total_bw[1])}"

            print(f"id: {self.id} ,Total cycles spent on input grad comm: {format_value(self.total_input_grad_comm / Sys.FREQ)}")
            total_bw = self.compute_busbw(self.input_grad_comm_type, input_grad_group_size, self.input_grad_comm_size, self.total_input_grad_comm)
            data = data + f",{format_value(self.total_input_grad_comm / Sys.FREQ)}"
            data = data + f",{format_value_bs(total_bw[0])}"
            data = data + f",{format_value_bs(total_bw[1])}"

            EndToEnd.write_line(data)

            if layer_num == self.workload.SIZE - 1:
                if param.mode != 'ANALYTICAL':
                    total_exposed = (Sys.boostedTick() / Sys.FREQ) - total_compute
                Expose_PP_time = (2 * vpp * GA * (pp_commsize * 1e9 / param.net_work_param.pp) / Sys.FREQ)
                Expose_PP_time *= (1 - param.net_work_param.pp_overlap_ratio)
                pre_bubble_time *= (PP_size - 1) / (GA * vpp)
                total_time = total_compute + total_exposed + pre_bubble_time + Expose_PP_time

                def format_percentage(value):
                    percentage = (value / total_time) * 100
                    return f"{percentage:.2f}%"

                file_name = param.res
                last_slash_pos = param.res.rfind('/')
                if last_slash_pos != -1:
                    file_name = param.res[last_slash_pos + 1:]

                keys = "File name, Expose DP comm, Expose DP_EP comm, Expose TP comm, Expose_EP_comm, Expose_PP_comm, bubble time, total comp, total exposed comm, Total time"
                values = f"{file_name}, {format_value(DP_comm)} ({format_percentage(DP_comm)}), {format_value(DP_EP_comm)} ({format_percentage(DP_EP_comm)}), {format_value(Expose_TP_comm)} ({format_percentage(Expose_TP_comm)}), {format_value(Expose_EP_comm)} ({format_percentage(Expose_EP_comm)}), {format_value(Expose_PP_time)} ({format_percentage(Expose_PP_time)}), {format_value(pre_bubble_time)} ({format_percentage(pre_bubble_time)}), {format_value(total_compute)} ({format_percentage(total_compute)}), {format_value(total_exposed)} ({format_percentage(total_exposed)}), {format_value(total_time)}"
                data = keys + "\n" + values
                EndToEnd.write_res(data)

                if param.net_work_param.visual:
                    chart_path = EndToEnd.path
                    with open(chart_path + "chart.html", 'w') as htmlFile:
                        htmlFile.write("<!DOCTYPE html>\n")
                        htmlFile.write("<html>\n<head>\n")
                        htmlFile.write("<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n")
                        htmlFile.write("<style>\n")
                        htmlFile.write("body { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 50vh; margin: 0; padding-top: 10%; }\n")
                        htmlFile.write("canvas { width: 50%; max-width: 400px; height: auto; }\n")
                        htmlFile.write("h2 { margin: 5px 0; }\n")
                        htmlFile.write("</style>\n")
                        htmlFile.write("</head>\n<body>\n")
                        htmlFile.write("<canvas id=\"myPieChart\"></canvas>\n")
                        htmlFile.write(f"<h2>Total Time: {total_time} ns</h2>\n")
                        htmlFile.write(f"<h2>model: {file_name} </h2>\n")
                        htmlFile.write("<script>\n")
                        htmlFile.write("var ctx = document.getElementById('myPieChart').getContext('2d');\n")
                        htmlFile.write("var myPieChart = new Chart(ctx, {\n")
                        htmlFile.write("    type: 'pie',\n")
                        htmlFile.write("    data: {\n")
                        htmlFile.write("        labels: ['Expose DP comm', 'Expose DP_EP comm','Expose TP comm', 'Expose_EP_comm','Total compute', 'PP Bubble time', 'Expose PP comm'],\n")
                        htmlFile.write("        datasets: [{\n")
                        htmlFile.write(f"            data: [{DP_comm}, {DP_EP_comm}, {Expose_TP_comm}, {Expose_EP_comm}, {total_compute}, {pre_bubble_time}, {Expose_PP_time}],\n")
                        htmlFile.write("            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40','#FF5733'],\n")
                        htmlFile.write("        }]\n")
                        htmlFile.write("    },\n")
                        htmlFile.write("    options: {\n")
                        htmlFile.write("        responsive: true,\n")
                        htmlFile.write("        maintainAspectRatio: true,\n")
                        htmlFile.write("        plugins: {\n")
                        htmlFile.write("            tooltip: {\n")
                        htmlFile.write("                callbacks: {\n")
                        htmlFile.write("                    label: function(context) {\n")
                        htmlFile.write("                        var label = context.label || '';\n")
                        htmlFile.write("                        if (label) {\n")
                        htmlFile.write("                            label += ': ';\n")
                        htmlFile.write("                        }\n")
                        htmlFile.write("                        if (context.parsed !== null) {\n")
                        htmlFile.write("                            label += context.parsed + ' ns';\n")
                        htmlFile.write("                        }\n")
                        htmlFile.write("                        return label;\n")
                        htmlFile.write("                    }\n")
                        htmlFile.write("                }\n")
                        htmlFile.write("            }\n")
                        htmlFile.write("        }\n")
                        htmlFile.write("    }\n")
                        htmlFile.write("});\n")
                        htmlFile.write("</script>\n")
                        htmlFile.write("</body>\n</html>")
                    print("HTML file created")

        return layerData

    def take_stream_stats_average(self):
        pass

    def update_stream_stats(self, dataset):
        pass

    def compute_time(
        self,
        comtype,
        tp_size,
        nranks,
        data_size,
        group_type,
        all_gpus,
        ep_size
    ):
        param = UserParam.getInstance()
        comp_time = 0
        if comtype == ComType.None:
            return 0

        DP_comm_inside = False
        TP_comm_inside = False
        EP_comm_inside = False
        n_ranks = 0
        nnics = 0
        gpus_per_server = param.net_work_param.gpus_per_server
        gpu_type = param.net_work_param.gpu_type
        tp_ar = param.net_work_param.tp_ar
        tp_ag = param.net_work_param.tp_ag
        tp_ata = param.net_work_param.tp_ata
        ep_ata = param.net_work_param.ep_ata
        dp_ag = param.net_work_param.dp_ag
        ep_ag = param.net_work_param.ep_ag
        dp_ar = param.net_work_param.dp_ar
        ep_ar = param.net_work_param.ep_ar

        if group_type == MockNccl.GroupType.TP or group_type == MockNccl.GroupType.EP:
            n_ranks = tp_size
            if n_ranks <= gpus_per_server:
                TP_comm_inside = True
        elif group_type in [MockNccl.GroupType.DP, MockNccl.GroupType.EP, MockNccl.GroupType.DP_EP]:
            n_ranks = nranks
            nnics = gpus_per_server // tp_size
            if all_gpus == gpus_per_server and tp_size <= gpus_per_server:
                DP_comm_inside = True

        if TP_comm_inside or DP_comm_inside:
            if comtype == ComType.All_Reduce:
                comp_time = data_size * 1e9 / tp_ar * 2 * (nranks - 1) / (nranks / 1.0)
            elif group_type == MockNccl.GroupType.TP and comtype in [ComType.All_Gather, ComType.Reduce_Scatter]:
                comp_time = data_size * 1e9 / tp_ag * (nranks - 1) / (nranks / 1.0)
            elif group_type == MockNccl.GroupType.TP and comtype == ComType.All_to_All:
                comp_time = data_size * 1e9 / tp_ata * (nranks - 1) / (nranks / 1.0)
            elif group_type == MockNccl.GroupType.EP and comtype == ComType.All_to_All:
                comp_time = data_size * 1e9 / ep_ata * (nranks - 1) / (nranks / 1.0)
            else:
                comp_time = 0
        elif not TP_comm_inside and group_type == MockNccl.GroupType.TP:
            if comtype == ComType.All_Reduce:
                comp_time = data_size * 1e9 / tp_ar * 2 * (nranks - 1) / (nranks / 1.0)
            elif comtype in [ComType.All_Gather, ComType