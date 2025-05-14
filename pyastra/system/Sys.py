import math
import time
import os
import re
import sys
import enum
import threading
import collections
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from datetime import datetime

from topology.ComplexLogicalTopology import *
from system.topology import GeneralComplexTopology, \
  LocalRingGlobalBinaryTree, LocalRingNodeA2AGlobalDBT, \
  BasicLogicalTopology, DoubleBinaryTreeTopology, Torus3d, LogicalTopology
from BaseStream import * 
from DataSet import * 
from MemBus import * 
from QueueLevels import *
from SimRecvCaller import * 
from SimSendCaller import *
from StreamBaseline import * 
from workload.Workload import LoopState
import Common
from RendezvousRecvData import * 
from RendezvousSendData import *
from system.collective import AllToAll, \
  DoubleBinaryTreeAllReduce, HalvingDoubling, Ring, NcclTreeFlowModel
from system.scheduling import OfflineGreedy
from MockNcclLog import * 
from workload.Layer import *
from AstraMemoryAPI import * 
from AstraNetworkAPI import *
from Callable import *
from CollectivePhase import *
from SendPacketEventHandlerData import *
from UsageTracker import *
from MockNcclChannel import *
from BasicEventHandlerData import *

class ParallelStrategy(enum.Enum):
    TP = 0
    DP = 1
    PP = 2
    EP = 3
    DP_EP = 4
    NONE = 5

class SchedulerUnit:
  def __init__(self, sys, queues, max_running_streams, ready_list_threshold, queue_threshold):
    self.sys = sys
    self.ready_list_threshold = ready_list_threshold
    self.queue_threshold = queue_threshold
    self.max_running_streams = max_running_streams
    
    self.latency_per_dimension = [0] * len(queues)
    self.total_chunks_per_dimension = [0] * len(queues)
    self.total_active_chunks_per_dimension = [0] * len(queues)
    
    base = 0
    dimension = 0
    
    self.running_streams = {}
    self.stream_pointer = {}
    self.queue_id_to_dimension = {}
    self.usage = []
    
    for q in queues:
        for i in range(q):
            self.running_streams[base] = 0
            # Python没有直接等价于C++迭代器的概念，使用None替代
            #TODO: 这里实现不完整, 后面需要立刻加上去
            self.stream_pointer[base] = None
            self.queue_id_to_dimension[base] = dimension
            base += 1
        
        dimension += 1
        self.usage.append(UsageTracker(2))

  def notify_stream_removed(self, vnet, running_time):
      dimension = self.queue_id_to_dimension[vnet]
      
      self.total_active_chunks_per_dimension[dimension] -= 1
      if self.sys.id == 0 and self.total_active_chunks_per_dimension[dimension] == 0:
          self.usage[dimension].decrease_usage()
      
      self.running_streams[vnet] -= 1
      
      self.latency_per_dimension[dimension] += running_time
      self.total_chunks_per_dimension[dimension] += 1
      
      if (self.sys.first_phase_streams < self.ready_list_threshold and
          self.sys.total_running_streams < self.max_running_streams):
          max_count = self.ready_list_threshold - self.sys.first_phase_streams
          if max_count > self.max_running_streams - self.sys.total_running_streams:
              max_count = self.max_running_streams - self.sys.total_running_streams
          self.sys.schedule(max_count)
      
      stream_list = self.sys.active_Streams[vnet]
      current_idx = self.running_streams[vnet]
      
      while current_idx < len(stream_list) and current_idx < self.queue_threshold:
          stream_list[current_idx].init()
          self.running_streams[vnet] += 1
          current_idx += 1

  def notify_stream_added(self, vnet):
      dimension = self.queue_id_to_dimension[vnet]
      
      self.total_active_chunks_per_dimension[dimension] += 1
      if self.sys.id == 0 and self.total_active_chunks_per_dimension[dimension] == 1:
          self.usage[dimension].increase_usage()
      
      stream_list = self.sys.active_Streams[vnet]
      
      start_idx = self.running_streams[vnet]
      
      for i in range(start_idx, min(len(stream_list), self.queue_threshold)):
          stream_list[i].init()
          self.running_streams[vnet] += 1
      
      nccl_log = MockNcclLog.getInstance()
      nccl_log.writeLog(NcclLogLevel.DEBUG, "Sys::SchedulerUnit::notify_stream_added finished")

  def notify_stream_added_into_ready_list(self):
      if (self.sys.first_phase_streams < self.ready_list_threshold and
          self.sys.total_running_streams < self.max_running_streams):
          max_count = self.ready_list_threshold - self.sys.first_phase_streams
          if max_count > self.max_running_streams - self.sys.total_running_streams:
              max_count = self.max_running_streams - self.sys.total_running_streams

          self.sys.schedule(max_count)
      return

  def get_average_latency_per_dimension(self):
      result = []
      for i in range(len(self.latency_per_dimension)):
          if self.total_chunks_per_dimension[i] > 0:
              avg = self.latency_per_dimension[i] / self.total_chunks_per_dimension[i]
          else:
              avg = -1  
          result.append(avg)
      return result

g_sys_inCriticalSection = threading.Lock()


class sysCriticalSection:
    def __init__(self):
        while g_sys_inCriticalSection.acquire():
            pass

    def ExitSection(self):
        g_sys_inCriticalSection.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ExitSection()

class Sys(Callable):
    offset = 0  
    all_generators = []  
    dummy_data = 2  

    def __init__(
        self,
        NI,
        MEM,
        id,
        npu_offset,
        num_passes,
        physical_dims,
        queues_per_dim,
        my_sys,
        my_workload,
        comm_scale,
        compute_scale,
        injection_scale,
        total_stat_rows,
        stat_row,
        path,
        run_name,
        seprate_log,
        rendezvous_enabled,
        gpu_type,
        all_gpus,
        NVSwitchs,
        ngpus_per_node
    ):
        self.NS3_MTP = int(os.getenv('NS3_MTP', 0)) 
        self.PHY_MTP = int(os.getenv('PHY_MTP', 0))

        self.scheduler_unit = None
        self.vLevels = None
        self.memBus = None
        self.workload = None
        self.offline_greedy = None
        
        self.initialized = False
        self.intra_dimension_scheduling = IntraDimensionScheduling.FIFO
        self.inter_dimension_scheduling = InterDimensionScheduling.Ascending
        self.round_robin_inter_dimension_scheduler = 0
        self.last_scheduled_collective = 0
        self.dim_to_break = -1
        
        self.start_sim_time = datetime.datetime.now()
        
        self.NI = NI
        self.MEM = MEM
        self.id = id
        self.npu_offset = npu_offset
        self.method = "baseline"
        self.finished_workloads = 0
        self.streams_finished = 0
        self.streams_injected = 0
        self.first_phase_streams = 0
        self.total_running_streams = 0
        self.priority_counter = 0
        self.comm_scale = comm_scale
        self.compute_scale = compute_scale
        self.injection_scale = injection_scale
        self.inp_model_shared_bus = 0
        self.inp_boost_mode = 0
        self.num_channels = 1
        self.processing_latency = 10
        self.communication_delay = 10
        self.local_reduction_delay = 1
        self.active_chunks_per_dimension = 1
        self.seprate_log = seprate_log
        self.rendezvous_enabled = rendezvous_enabled
        self.NVSwitchs = NVSwitchs
        self.all_gpus = all_gpus
        self.gpu_type = gpu_type
        self.ngpus_per_node = ngpus_per_node
        
        if (id + npu_offset + 1) > len(self.all_generators):
            self.all_generators.extend([None] * ((id + npu_offset + 1) - len(self.all_generators)))
        self.all_generators[id + npu_offset] = self
        
        self.inp_scheduling_policy = "LIFO"
        self.communication_delay = 10 * injection_scale
        self.active_chunks_per_dimension = 1
        self.preferred_dataset_splits = 1
        self.inp_boost_mode = 0
        self.inp_all_reduce_implementation = "NcclFlowModel"
        self.inp_all_gather_implementation = "NcclFlowModel"
        self.inp_reduce_scatter_implementation = "NcclFlowModel"
        self.inp_all_to_all_implementation = "NcclFlowModel"
        self.inp_collective_optimization = "baseline"
        
        self.registered_for_finished_stream_event = []

        result = self.post_process_inputs()
        if not result:
            self.sys_panic("Unable to initialize the system layer because the file can not be openned")
        
        self.pending_events = 0
        
        total_disabled = 0
        self.physical_dims = physical_dims
        self.queues_per_dim = queues_per_dim
        element = 0
        self.all_queues = 0
        self.total_nodes = 1
        
        for current_dim in range(len(queues_per_dim)):
            self.all_queues += queues_per_dim[current_dim]
            enabled = not self.boost_mode
            
            if id % self.total_nodes == 0 and id < self.total_nodes * physical_dims[current_dim]:
                enabled = True
                
            if not enabled:
                total_disabled += queues_per_dim[current_dim]
                
            if physical_dims[current_dim] >= 1:
                self.total_nodes *= physical_dims[current_dim]
                
            for j in range(queues_per_dim[current_dim]):
                self.active_Streams[element] = []  
                self.stream_priorities[element] = []  
                element += 1
        
        if self.all_queues == total_disabled:
            self.NI.enabled = False
            print(f"Node {id} has been totally disabled")
        
        self.concurrent_streams = int(math.ceil(float(self.active_chunks_per_dimension) / queues_per_dim[0]))
        self.active_first_phase = 100000000
        
        if id == 0:
            print(f"The final active chunks per dimension 1 after allocating to queues is: {self.concurrent_streams * queues_per_dim[0]}")
        
        self.max_running = 100000000
        
        self.scheduler_unit = SchedulerUnit(
            self,
            queues_per_dim,
            self.max_running,
            self.active_first_phase,
            self.concurrent_streams
        )
        self.vLevels = QueueLevels(queues_per_dim, 0, self.NI.get_backend_type())
        
        self.logical_topologies = {
            "AllReduce": GeneralComplexTopology(id, physical_dims, self.all_reduce_implementation_per_dimension),
            "ReduceScatter": GeneralComplexTopology(id, physical_dims, self.reduce_scatter_implementation_per_dimension),
            "AllGather": GeneralComplexTopology(id, physical_dims, self.all_gather_implementation_per_dimension),
            "AllToAll": GeneralComplexTopology(id, physical_dims, self.all_to_all_implementation_per_dimension)
        }
        
        self.stream_counter = 0
        
        if id == 0:
            atexit.register(self.exiting)
            print(f"total nodes: {self.total_nodes}")
        
        self.running_list = []
        self.ready_list = []
        self.NI.sim_init(self.MEM)
        self.memBus = MemBus(
            "NPU",
            "MA",
            self,
            self.inp_L,
            self.inp_o,
            self.inp_g,
            self.inp_G,
            self.model_shared_bus,
            self.communication_delay,
            True
        )
        
        # 初始化工作负载
        self.workload = Workload(
            run_name,
            self,
            my_workload,
            num_passes,
            total_stat_rows,
            stat_row,
            path,
            self.seprate_log
        )
        
        if not self.workload.initialized:
            self.sys_panic("Unable to initialize the workload layer because it can not open the workload file")
            return
        
        # 条件初始化（对应C++中的#ifdef）
        if hasattr(self, 'NS3_MTP') or hasattr(self, 'NS3_MPI') or hasattr(self, 'PHY_MTP'):
            result = self.mock_nccl_grobal_group_init()
            if not result:
                self.sys_panic("Unable to initialize the system grobal group because the file can not be openned")
            
            result = self.mock_nccl_comms_init()
            if not result:
                self.sys_panic("Unable to initialize the system mockncclComm because the file can not be openned")
        
        # 离线调度优化器
        if (self.inter_dimension_scheduling == InterDimensionScheduling.OfflineGreedy or
            self.inter_dimension_scheduling == InterDimensionScheduling.OfflineGreedyFlex):
            self.offline_greedy = OfflineGreedy(self)
        
        self.initialized = True

    def __del__(self):
      self.end_sim_time = datetime.datetime.now()
      duration = (self.end_sim_time - self.start_sim_time).total_seconds() // 60
      
      if self.id == 0:
          timenow = datetime.datetime.now()
          print("*****")
          print(f"Time to exit: {timenow.strftime('%c')}")
          print(f"all-reduce Collective implementation: {self.inp_all_reduce_implementation}")
          print(f"reduce-scatter Collective implementation: {self.inp_reduce_scatter_implementation}")
          print(f"all-gather Collective implementation: {self.inp_all_gather_implementation}")
          print(f"all-to-all Collective implementation: {self.inp_all_to_all_implementation}")
          print(f"Collective optimization: {self.inp_collective_optimization}")
          print(f"Total sim duration: {int(duration // 60)}:{int(duration % 60)} hours")
          print(f"Total streams injected: {self.streams_injected}")
          print(f"Total streams finished: {self.streams_finished}")
          print(f"Percentage of finished streams: {(self.streams_finished / self.streams_injected) * 100} %")
          print("*****")
      
      # 非PHY_MTP模式下的清理操作
      if not getattr(self, 'PHY_MTP', False):
          self.all_generators[self.id + self.npu_offset] = None
          
          for lt in self.logical_topologies.values():
              del lt
          self.logical_topologies.clear()
          
          for ci in self.all_reduce_implementation_per_dimension:
              del ci
          for ci in self.reduce_scatter_implementation_per_dimension:
              del ci
          for ci in self.all_gather_implementation_per_dimension:
              del ci
          for ci in self.all_to_all_implementation_per_dimension:
              del ci
          
          if self.scheduler_unit:
              del self.scheduler_unit
          if self.vLevels:
              del self.vLevels
          if self.memBus:
              del self.memBus
          if self.workload:
              del self.workload
          if self.offline_greedy:
              del self.offline_greedy
          
          should_exit = True
          for i in range(self.num_gpus):
              if self.all_generators[i] is not None:
                  should_exit = False
                  break
          
          if should_exit:
              self.exitSimLoop("Exiting")
      else:
          self.exitSimLoop("Exiting")

    def register_for_finished_stream(self, callable_obj):
        self.registered_for_finished_stream_event.append(callable_obj)

    def increase_finished_streams(self, amount):
        self.streams_finished += amount
        for c in self.registered_for_finished_stream_event:
            c.call(Common.EventType.StreamsFinishedIncrease, None)

    def zero_latecy_register_event(self, callable_obj, event_type, call_data, cycles):
        mycycles = 0
        should_schedule = False

        cs = None

        if self.NS3_MTP or self.PHY_MTP:
          cs = sysCriticalSection()

        if self.boosted_tick() + mycycles not in self.event_queue:
            self.event_queue[self.boosted_tick() + mycycles] = []
            should_schedule = True

        self.event_queue[self.boosted_tick() + mycycles].append((callable_obj, event_type, call_data))
        self.pending_events += 1

        if should_schedule:
            tmp = self.generate_time(mycycles)
            data = BasicEventHandlerData(self, EventType.CallEvents)
            self.handleEvent(data)

        if self.NS3_MTP or self.PHY_MTP:
          cs.ExitSection()

    def register_event(
        self,
        callable_obj,
        event,
        callData,
        cycles: int
    ):
      mycycles = cycles  # 
      self.try_register_event(callable_obj, event, callData, mycycles)

    def insert_into_ready_list(self, stream: BaseStream):
        self.insert_stream(self.ready_list, stream)
        self.scheduler_unit.notify_stream_added_into_ready_list()

    def insert_into_running_list(self, stream):
      self.running_list.append(stream)

    def schedule(self, num: int) -> None:
        nccl_log = MockNcclLog.getInstance()
        ready_list_size = len(self.ready_list)
        counter = min(num, ready_list_size)
        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, 
                         f"Sys.cc::schedule num {num} ready_list_size {ready_list_size}")

        while counter > 0:
            current_stream = self.ready_list[0]  
            top_vn = current_stream.phases_to_go[0].queue_id  
            total_waiting_streams = len(self.ready_list)     
            total_phases = len(current_stream.phases_to_go)    

            # 推进流到下一个虚拟网络阶段（基线版本）
            self.proceed_to_next_vnet_baseline(current_stream)

            if self.PHY_MTP: 
                if current_stream.current_queue_id == -1:
                    self.sys_panic("should not happen!")
                
                self.ready_list.pop(0)
                self.first_phase_streams += 1
                self.total_running_streams += 1

            counter -= 1

        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "Sys::schedule finished")

    def register_phases(
        self,
        stream: BaseStream,
        phases_to_go: list[CollectivePhase]
    ) -> None:
        for vnet in phases_to_go:
            self.stream_priorities[vnet.queue_id].append(stream.stream_num)

    def call(self, event_type, data):
      if self.id == 0 and event_type == Common.EventType.General:
        self.increase_finished_streams(1)

    def try_register_event(
        self,
        callable_obj: Callable,
        event: EventType,
        call_data: CallData,
        cycles: int  
    ) -> None:
        should_schedule = False
        my_cycles = cycles 

        nccl_log = MockNcclLog.getInstance()
        nccl_log.writeLog(NcclLogLevel.DEBUG, f"try_register_event EventType {event}")

        cs = None
        if self.NS3_MTP:
            cs = sysCriticalSection()

        current_tick = self.boosted_tick()
        target_tick = current_tick + my_cycles

        if target_tick not in self.event_queue:
            self.event_queue[target_tick] = []
            should_schedule = True

        self.event_queue[target_tick].append((callable_obj, event, call_data))

        if self.NS3_MTP:
          cs.ExitSection()

        if should_schedule:
            tmp = self.generate_time(my_cycles)
            data = BasicEventHandlerData(self, Common.EventType.CallEvents)
            self.NI.sim_schedule(tmp, self.handleEvent, data) 

        cycles = 0  
        self.pending_events += 1

    def call_events(self) -> None:
        current_tick = self.boostedTick()  
        
        if current_tick not in self.event_queue:
            finish_condition = (
                (self.finished_workloads == 1 and 
                 len(self.event_queue) == 0 and 
                 len(self.pending_sends) == 0) or 
                not self.initialized
            )
            if finish_condition:
                pass
            return
        
        # 执行事件队列中的可调用对象
        callables: List[Tuple[Callable, Any, Any]] = self.event_queue[current_tick]
        for callable_entry in callables:
            try:
                self.pending_events -= 1
                target, arg1, arg2 = callable_entry  # 解包元组
                target.call(arg1, arg2)  # 调用可调用对象
            except Exception as e:
                print("warning! a callable is removed before call", file=sys.stderr)
        
        # 线程安全地清除事件队列条目（假设sysCriticalSection是上下文管理器）
        
        cs = sysCriticalSection()
        if current_tick in self.event_queue:
          del self.event_queue[current_tick]        
        cs.ExitSection()

        # 执行FINISH_CHECK逻辑
        finish_condition = (
            (self.finished_workloads == 1 and 
             len(self.event_queue) == 0 and 
             len(self.pending_sends) == 0) or 
            not self.initialized
        )
        if finish_condition:
            pass

    def workload_finished(self):
      pass

    @staticmethod
    def boostedTick():
        """计算增强时间戳（基于全局生成器的模拟时间）"""
        ts = None
        if Sys.all_generators:  # 检查all_generators是否非空
            ts = Sys.all_generators[0]  # 初始取第一个元素
            # 如果第一个元素为空，遍历后续元素寻找非空实例
            if ts is None:
                for generator in Sys.all_generators[1:]:
                    if generator is not None:
                        ts = generator
                        break
        
        if ts is None:
            raise RuntimeError("No valid Sys instance found in all_generators")
        
        # 步骤2：获取模拟时间并计算tick
        tmp = ts.NI.sim_get_time()  # 调用NI模块的时间获取方法（假设NI是成员对象）
        tick = tmp.time_val // Common.CLOCK_PERIOD  # 整数除法（与C++行为一致）
        
        # 步骤3：加上偏移量并返回（假设offset是实例变量）
        return tick + Sys.offset


    @staticmethod
    def exiting():
        pass 

    def nextPowerOf2(self, n: int) -> int:
        if n <= 0:
            return 1  # 处理n=0或负数，返回1（2^0）
        if (n & (n - 1)) == 0:
            return n  # n已是2的幂
        count = 0
        while n != 0:
            n >>= 1
            count += 1
        return 1 << count

    @staticmethod
    def sys_panic(msg: str):
        print(msg, file=sys.stderr)
        sys.exit(1)

    def exitSimLoop(self, msg: str):
        if self.id == 0:
            print(msg)
        # 调用模拟结束接口（假设NI是类的成员对象）
        self.NI.sim_finish()


    def iterate(self):
        self.call_events() 

    def initialize_sys(self, name: str) -> bool:
        in_file = None
        try:
            in_file = open(name, 'r')
        except IOError:
            if self.id == 0:
                print(f"Unable to open file: {name}", file=sys.stderr)
                print("############ Exiting because unable to open the system input file ############", file=sys.stderr)
                print("This error is fatal. Please check your path and filename.", file=sys.stderr)
            sys.exit(1)
        
        if self.id == 0:
            print("Success in opening system file")
        
        while True:
            # 读取变量（跳过空白符，类似 C++ 的 inFile >> var）
            in_file.seek(in_file.tell())  # 定位当前位置
            if in_file.peek() == -1:  # EOF
                break
            
            # 读取变量（按空格分割，处理连续空白）
            tokens = in_file.readline().strip().split()
            if not tokens:
                continue  # 跳过空行
            
            var = tokens[0]
            value = tokens[1] if len(tokens) > 1 else ""
            
            result = self.parse_var(var, value)
            if not result:
                in_file.close()
                return result
        
        in_file.close()
        return self.post_process_inputs()

    def trim(
        input_str: str,
        whitespace: str = " \t"
    ) -> str:
        """去除字符串两端指定的空白字符（默认去除空格和制表符）
        
        Args:
            input_str: 待处理的输入字符串
            whitespace: 需要去除的空白字符集合（默认包含空格和制表符）
            
        Returns:
            去除两端空白后的新字符串（若全为空白则返回空字符串）
        """
        # 查找第一个非空白字符的索引
        str_begin = None
        for i, char in enumerate(input_str):
            if char not in whitespace:
                str_begin = i
                break
        if str_begin is None:  # 所有字符都是空白
            return ""
        
        # 查找最后一个非空白字符的索引
        str_end = None
        for i in reversed(range(len(input_str))):
            if input_str[i] not in whitespace:
                str_end = i
                break
        
        # 截取非空白子串（Python 切片左闭右开，需+1包含 str_end）
        return input_str[str_begin : str_end + 1]

    def parse_var(self, var: str, value: str) -> bool:
        var = self.trim(var)
        value = self.trim(value)
        if self.id == 0:
            print(f"Var is: {var}, val is: {value}")

        if var == "scheduling-policy:":
            self.inp_scheduling_policy = value
        elif var == "all-reduce-implementation:":
            self.inp_all_reduce_implementation = value
        elif var == "reduce-scatter-implementation:":
            self.inp_reduce_scatter_implementation = value
        elif var == "all-gather-implementation:":
            self.inp_all_gather_implementation = value
        elif var == "all-to-all-implementation:":
            self.inp_all_to_all_implementation = value
        elif var == "collective-optimization:":
            self.inp_collective_optimization = value
        elif var == "endpoint-delay:":
            self.communication_delay = int(value)
            self.communication_delay = self.communication_delay * self.injection_scale
        elif var == "local-reduction-delay:":
            self.local_reduction_delay = int(value)
        elif var == "active-chunks-per-dimension:":
            self.active_chunks_per_dimension = int(value)
        elif var == "L:":
            self.inp_L = int(value)
        elif var == "o:":
            self.inp_o = int(value)
        elif var == "g:":
            self.inp_g = int(value)
        elif var == "G:":
            self.inp_G = int(value)
        elif var == "model-shared-bus:":
            self.inp_model_shared_bus = int(value)
        elif var == "preferred-dataset-splits:":
            self.preferred_dataset_splits = int(value)
        elif var == "boost-mode:":
            self.inp_boost_mode = int(value)
        elif var == "intra-dimension-scheduling:":
            if value == "FIFO":
                self.intra_dimension_scheduling = IntraDimensionScheduling.FIFO
            elif value == "RG":
                self.intra_dimension_scheduling = IntraDimensionScheduling.RG
            elif value == "smallestFirst":
                self.intra_dimension_scheduling = IntraDimensionScheduling.SmallestFirst
            elif value == "lessRemainingPhaseFirst":
                self.intra_dimension_scheduling = IntraDimensionScheduling.LessRemainingPhaseFirst
            else:
                raise ValueError("unknown value for intra-dimension-scheduling in sys input file")
        elif var == "inter-dimension-scheduling:":
            if value == "ascending":
                self.inter_dimension_scheduling = InterDimensionScheduling.Ascending
            elif value == "offlineGreedy":
                self.inter_dimension_scheduling = InterDimensionScheduling.OfflineGreedy
            elif value == "offlineGreedyFlex":
                self.inter_dimension_scheduling = InterDimensionScheduling.OfflineGreedyFlex
            elif value == "roundRobin":
                self.inter_dimension_scheduling = InterDimensionScheduling.RoundRobin
            else:
                raise ValueError("unknown value for inter-dimension-scheduling in sys input file")
        elif var == "seprate-log:":
            if int(value) == 0:
                self.seprate_log = False
            else:
                self.seprate_log = True
        elif var != "":
            print(f"######### Exiting because {var} is an unknown variable. Check your system input file. #########")
            exit(1)

        return True

    def post_process_inputs(self) -> bool:
        self.all_reduce_implementation_per_dimension = self.generate_collective_implementation_from_input(
            self.inp_all_reduce_implementation)
        if len(self.all_reduce_implementation_per_dimension) == 0:
            raise ValueError("unknown value for all-reduce-implementation in sys input file")

        self.reduce_scatter_implementation_per_dimension = self.generate_collective_implementation_from_input(
            self.inp_reduce_scatter_implementation)
        if len(self.reduce_scatter_implementation_per_dimension) == 0:
            raise ValueError("unknown value for reduce-scatter-implementation in sys input file")

        self.all_gather_implementation_per_dimension = self.generate_collective_implementation_from_input(
            self.inp_all_gather_implementation)
        if len(self.all_gather_implementation_per_dimension) == 0:
            raise ValueError("unknown value for all-gather-implementation in sys input file")

        self.all_to_all_implementation_per_dimension = self.generate_collective_implementation_from_input(
            self.inp_all_to_all_implementation)
        if len(self.all_to_all_implementation_per_dimension) == 0:
            raise ValueError("unknown value for all-to-all-implementation in sys input file")

        if self.inp_collective_optimization == "baseline":
            self.collectiveOptimization = CollectiveOptimization.Baseline
        elif self.inp_collective_optimization == "localBWAware":
            self.collectiveOptimization = CollectiveOptimization.LocalBWAware
        else:
            raise ValueError("unknown value for collective optimization in sys input file")

        if self.inp_boost_mode == 1:
            self.boost_mode = True
        else:
            self.boost_mode = False

        if self.inp_scheduling_policy == "LIFO":
            self.scheduling_policy = SchedulingPolicy.LIFO
        elif self.inp_scheduling_policy == "FIFO":
            self.scheduling_policy = SchedulingPolicy.FIFO
        else:
            raise ValueError("unknown value for scheduling policy in sys input file")

        if self.inp_model_shared_bus == 1:
            self.model_shared_bus = True
        else:
            self.model_shared_bus = False

        return True

    def generate_collective_implementation_from_input(input_str: str) -> List[CollectiveImplementation]:
        """根据输入字符串生成集合通信实现列表
        
        Args:
            input_str: 输入字符串（格式示例："ring_direct5_oneRing"）
            
        Returns:
            集合通信实现对象列表
            
        Raises:
            ValueError: 无法解析输入字符串时抛出
        """
        # 按下划线分割输入字符串（对应 C++ 的 split_string(input, "_")）
        inputs_per_dimension = input_str.split("_")
        result: List[CollectiveImplementation] = []

        for dimension_input in inputs_per_dimension:
            if dimension_input == "ring":
                result.append(CollectiveImplementation(CollectiveImplementationType.Ring))
            elif dimension_input == "oneRing":
                result.append(CollectiveImplementation(CollectiveImplementationType.OneRing))
            elif dimension_input == "doubleBinaryTree":
                result.append(CollectiveImplementation(CollectiveImplementationType.DoubleBinaryTree))
            elif dimension_input.startswith("direct"):
                # 提取窗口大小（示例："direct5" → window=5）
                window: Optional[int] = -1
                if dimension_input != "direct":
                    # C++ substr(6,5) 对应 Python [6:11]（取从第6位开始的最多5个字符）
                    window_str = dimension_input[6:11]
                    window = int(window_str) if window_str.isdigit() else -1
                result.append(DirectCollectiveImplementation(CollectiveImplementationType.Direct, window))
            elif dimension_input.startswith("oneDirect"):
                # 提取窗口大小（示例："oneDirect10" → window=10）
                window: Optional[int] = -1
                if dimension_input != "oneDirect":
                    # C++ substr(9,5) 对应 Python [9:14]
                    window_str = dimension_input[9:14]
                    window = int(window_str) if window_str.isdigit() else -1
                result.append(DirectCollectiveImplementation(CollectiveImplementationType.OneDirect, window))
            elif dimension_input == "halvingDoubling":
                result.append(CollectiveImplementation(CollectiveImplementationType.HalvingDoubling))
            elif dimension_input == "oneHalvingDoubling":
                result.append(CollectiveImplementation(CollectiveImplementationType.OneHalvingDoubling))
            elif dimension_input == "NcclFlowModel":
                result.append(CollectiveImplementation(CollectiveImplementationType.NcclFlowModel))
            elif dimension_input == "ncclRingTreeModel":
                result.append(CollectiveImplementation(CollectiveImplementationType.NcclTreeFlowModel))
            else:
                raise ValueError(
                    "Cannot interpret collective implementations. Please check the collective implementations in the sys input file"
                )

        return result

    def break_dimension(self, model_parallel_npu_group):
        if model_parallel_npu_group == 1:
            return -1
        
        dimension_to_break = 0
        all_npus = 1
        
        for dimension_to_break in range(len(self.physical_dims)):
            if all_npus * self.physical_dims[dimension_to_break] < model_parallel_npu_group:
                all_npus *= self.physical_dims[dimension_to_break]
            elif all_npus * self.physical_dims[dimension_to_break] > model_parallel_npu_group:
                for lt in self.logical_topologies.values():
                    del lt
                self.logical_topologies.clear()
                
                del self.scheduler_unit
                del self.vLevels
                
                self.queues_per_dim.insert(dimension_to_break, self.queues_per_dim[dimension_to_break])
                
                self.scheduler_unit = SchedulerUnit(
                    self,
                    self.queues_per_dim,
                    self.max_running,
                    self.active_first_phase,
                    self.concurrent_streams
                )
                self.vLevels = QueueLevels(self.queues_per_dim, 0, self.NI.get_backend_type())
                
                first_subdim = model_parallel_npu_group // all_npus
                second_subdim = self.physical_dims[dimension_to_break] // first_subdim
                
                logical_dims = []
                for dim in range(len(self.physical_dims)):
                    if dim != dimension_to_break:
                        logical_dims.append(self.physical_dims[dim])
                    else:
                        logical_dims.append(first_subdim)
                        logical_dims.append(second_subdim)
                
                def clone_and_insert(impl_list, dim_idx):
                    if len(impl_list) > dim_idx:
                        replicate = impl_list[dim_idx].clone()
                        impl_list.insert(dim_idx, replicate)
                    else:
                        replicate = impl_list[-1].clone() if impl_list else None
                        impl_list.append(replicate)
                    return impl_list
                
                self.all_reduce_implementation_per_dimension = clone_and_insert(
                    self.all_reduce_implementation_per_dimension, dimension_to_break)
                self.reduce_scatter_implementation_per_dimension = clone_and_insert(
                    self.reduce_scatter_implementation_per_dimension, dimension_to_break)
                self.all_gather_implementation_per_dimension = clone_and_insert(
                    self.all_gather_implementation_per_dimension, dimension_to_break)
                self.all_to_all_implementation_per_dimension = clone_and_insert(
                    self.all_to_all_implementation_per_dimension, dimension_to_break)
                
                # 重新创建逻辑拓扑
                self.logical_topologies = {
                    "AllReduce": GeneralComplexTopology(id, logical_dims, self.all_reduce_implementation_per_dimension),
                    "ReduceScatter": GeneralComplexTopology(id, logical_dims, self.reduce_scatter_implementation_per_dimension),
                    "AllGather": GeneralComplexTopology(id, logical_dims, self.all_gather_implementation_per_dimension),
                    "AllToAll": GeneralComplexTopology(id, logical_dims, self.all_to_all_implementation_per_dimension)
                }
                
                # 记录逻辑分割维度
                self.logical_broken_dims = logical_dims
                self.dim_to_break = dimension_to_break
                
                return dimension_to_break
            elif all_npus * self.physical_dims[dimension_to_break] == model_parallel_npu_group:
                return dimension_to_break
        
        return -1

    def front_end_sim_send(
        self,
        delay: int,
        buffer: object,
        count: int,
        data_type: int,
        dst: int,
        tag: int,
        request: sim_request,
        msg_handler,
        fun_arg
    ) -> int:
        if self.rendezvous_enabled:
            return self.rendezvous_sim_send(
                delay, buffer, count, data_type, dst, tag, request, msg_handler, fun_arg
            )
        else:
            return self.sim_send(
                delay, buffer, count, data_type, dst, tag, request, msg_handler, fun_arg
            )

    def front_end_sim_recv(
        self,
        delay: int,
        buffer: object,
        count: int,
        type: int,
        src: int,
        tag: int,
        request: sim_request,
        msg_handler,
        fun_arg
    ) -> int:
        if self.rendezvous_enabled:
            return self.rendezvous_sim_recv(
                delay, buffer, count, type, src, tag, request, msg_handler, fun_arg
            )
        else:
            return self.sim_recv(
                delay, buffer, count, type, src, tag, request, msg_handler, fun_arg
            )

    def sim_send(
        self,
        delay: int,
        buffer: object,
        count: int,
        data_type: int,
        dst: int,
        tag: int,
        request: 'sim_request',
        msg_handler: callable = None,
        fun_arg: object = None
    ) -> int:
      if delay == 0 and fun_arg is None:
          cs = sysCriticalSection()

          fun_arg_tmp = SendPacketEventHandlerData(
              self, 
              self.id + self.npu_offset, 
              dst, 
              tag
          )
          fun_arg = fun_arg_tmp

          key = (dst, tag)

          if key not in self.is_there_pending_sends or not self.is_there_pending_sends[key]:
              self.is_there_pending_sends[key] = True

              cs.ExitSection()
          else:
              if key not in self.pending_sends:
                  self.pending_sends[key] = []

              caller = SimSendCaller(
                      self,
                      buffer,
                      count,
                      data_type,
                      dst,
                      tag,
                      request,
                      msg_handler,
                      fun_arg
                  )

              self.pending_sends[key].append(caller)

              cs.ExitSection()
              return 1

      if delay == 0:
          self.NI.sim_send(buffer, count, data_type, dst, tag, request, msg_handler, fun_arg)
      else:
          caller = SimSendCaller(
                  self,
                  buffer,
                  count,
                  data_type,
                  dst,
                  tag,
                  request,
                  msg_handler,
                  fun_arg
              )
          self.try_register_event(
              caller,
              Common.EventType.General,
              None,
              delay
          )

      return 1

    def sim_recv(
        self,
        delay: int,
        buffer: object,
        count: int,
        type: int,
        src: int,
        tag: int,
        request: sim_request,
        msg_handler: callable = None,
        fun_arg: object = None
    ) -> int:
        if delay == 0:
            self.NI.sim_recv(
                buffer,
                count,
                type,
                src,
                tag,
                request,
                msg_handler,
                fun_arg
            )
        else:
            self.try_register_event(
                SimRecvCaller(
                    self,               # 当前 Sys 实例（对应 C++ 的 this）
                    buffer,
                    count,
                    type,
                    src,
                    tag,
                    request,            # 传递 request 对象（对应 C++ 的 *request）
                    msg_handler,
                    fun_arg
                ),
                Common.EventType.General,      # 事件类型（需确保 EventType 枚举已定义）
                None,                   # 事件参数（原 C++ 中为 nullptr）
                delay                   # 延迟时间
            )
        return 1

    def rendezvous_sim_send(
        self,
        delay: int,
        buffer: Any,
        count: int,
        data_type: int,
        dst: int,
        tag: int,
        request: sim_request,
        msg_handler: Callable,
        fun_arg: Any
    ) -> int:

        rsd = RendezvousSendData(
            sys_id=self.id,
            sys_instance=self,
            buffer=buffer,
            count=count,
            data_type=data_type,
            dst=dst,
            tag=tag,
            request=request,
            msg_handler=msg_handler,
            fun_arg=fun_arg
        )

        new_req = sim_request(
            srcRank=request.dstRank,  # newReq.srcRank = request->dstRank
            dstRank=request.srcRank,  # newReq.dstRank = request->srcRank
            reqCount=8192,            # rendevouz_size固定为8192
            tag=tag + 500000000       # newTag = tag + 500000000
        )

        self.sim_recv(
            delay,
            buffer,
            8192,               # rendevouz_size
            data_type,
            dst,
            new_req.tag,
            new_req,
            Sys.handleEvent, 
            rsd
        )

        return 1

    def rendezvous_sim_recv(
        self,
        delay: int,
        buffer: object,
        count: int,
        data_type: int,
        src: int,
        tag: int,
        request: sim_request,
        msg_handler,
        fun_arg
    ) -> int:
        rrd = RendezvousRecvData(
            self.id,                  
            self,                     # 当前 Sys 实例（对应 C++ 的 this）
            buffer,
            count,
            data_type,
            src,
            tag,
            request,              
            msg_handler,
            fun_arg
        )

        new_req = request.clone()      
        rendevouz_size = 8192

        new_req.dstRank = request.srcRank  
        new_req.srcRank = request.dstRank 
        new_req.reqCount = rendevouz_size     
        new_tag = tag + 500000000
        new_req.tag = new_tag                

        self.sim_send(
            delay,
            buffer,
            rendevouz_size,    
            data_type,
            src,
            new_tag,
            new_req,           
            Sys.handleEvent,  
            rrd               
        )

        return 1

    def mem_read(self, bytes: int) -> int:
        """计算内存读取操作的延迟周期数
        
        Args:
            bytes: 读取的字节数（无符号整数）
            
        Returns:
            延迟的时钟周期数（整数）
        """
        # 处理 MEM 为空的情况（默认返回10周期）
        if self.MEM is None:
            return 10
        
        # 调用 MEM 模块的读取方法获取纳秒级延迟
        delay_ns = self.MEM.npu_mem_read(bytes)
        
        # 将纳秒延迟转换为时钟周期数（假设 CLOCK_PERIOD 是类的属性，表示每个周期的纳秒数）
        delay_cycles = delay_ns // self.CLOCK_PERIOD
        
        return delay_cycles

    def mem_write(self, bytes: int) -> int:
        """计算内存写入操作的延迟周期数
        
        Args:
            bytes: 写入的字节数（无符号整数）
            
        Returns:
            延迟的时钟周期数（整数）
        """
        # 处理 MEM 为空的情况（默认返回10周期）
        if self.MEM is None:
            return 10
        
        # 调用 MEM 模块的写入方法获取纳秒级延迟
        delay_ns = self.MEM.npu_mem_write(bytes)
        
        # 将纳秒延迟转换为时钟周期数（假设 CLOCK_PERIOD 是类的属性，表示每个周期的纳秒数）
        delay_cycles = delay_ns // self.CLOCK_PERIOD
        
        return delay_cycles

    @staticmethod
    def get_layer_numbers(workload_input: str) -> int:
      return Workload.get_layer_numbers(workload_input)

    def split_string(input_str: str, sep: str) -> list[str]:
        # 转义分隔符中的特殊字符，避免正则表达式解析错误
        escaped_sep = re.escape(sep)
        # 使用正则分割，保留连续分隔符之间的空串，后续过滤
        split_result = re.split(f'({escaped_sep})', input_str)
        
        # 过滤空串和分隔符本身，仅保留有效子串
        # 原 C++ 行为：strtok 会跳过连续分隔符，且不返回空串（包括首尾）
        result = []
        current = ''
        for token in split_result:
            if token == escaped_sep:
                if current:
                    result.append(current)
                    current = ''
            else:
                current += token
        if current:  # 处理末尾非分隔符的剩余字符
            result.append(current)
        
        return result

    def generate_all_reduce(
        self,
        size: int,
        involved_dimensions: list[bool],
        pref_scheduling: SchedulingPolicy,
        layer: int,
        event: Common.EventType,
        layer_ptr: Callable
    ) -> DataSet:
        """生成全归约操作的数据集（调用通用集合通信生成函数）
        
        Args:
            size: 数据总大小（无符号64位整数）
            involved_dimensions: 各维度是否参与的布尔列表
            pref_scheduling: 首选调度策略
            layer: 层次索引
            event: 事件类型
            layer_ptr: 层次指针（可调用对象）
            
        Returns:
            生成的数据集对象（DataSet实例）
        """
        return self.generate_collective(
            size,
            layer,
            self.logical_topologies["AllReduce"],
            self.all_reduce_implementation_per_dimension,
            involved_dimensions,
            ComType.All_Reduce,
            pref_scheduling,
            event,
            layer_ptr
        )

    def generate_all_to_all(
        self,
        size: int,
        involved_dimensions: list[bool],
        pref_scheduling: SchedulingPolicy,
        layer: int,
        event: Common.EventType,
        layer_ptr: Callable
    ) -> DataSet:
        """生成全对全通信操作的数据集（调用通用集合通信生成函数）
        
        Args:
            size: 数据总大小（无符号64位整数）
            involved_dimensions: 各维度是否参与的布尔列表（索引对应维度）
            pref_scheduling: 首选调度策略（如FIFO/LIFO等）
            layer: 层次索引（用于分层调度）
            event: 事件类型（触发后续操作的事件标识）
            layer_ptr: 层次指针（关联的可调用对象，用于事件回调）
            
        Returns:
            DataSet: 生成的全对全通信操作数据集对象
        """
        return self.generate_collective(
            size,
            layer,
            self.logical_topologies["AllToAll"],  # 逻辑拓扑：全对全对应项
            self.all_to_all_implementation_per_dimension,  # 全对全实现列表
            involved_dimensions,
            ComType.All_to_All,  # 通信类型：全对全
            pref_scheduling,
            event,
            layer_ptr
        )

    def generate_all_gather(
        self,
        size: int,
        involved_dimensions: list[bool],
        pref_scheduling: SchedulingPolicy,
        layer: int,
        event: Common.EventType,
        layer_ptr: Callable
    ) -> DataSet:
        """生成全收集操作的数据集（调用通用集合通信生成函数）
        
        Args:
            size: 数据总大小（无符号64位整数）
            involved_dimensions: 各维度是否参与的布尔列表
            pref_scheduling: 首选调度策略
            layer: 层次索引
            event: 事件类型
            layer_ptr: 层次指针（可调用对象）
            
        Returns:
            生成的数据集对象（DataSet实例）
        """
        return self.generate_collective(
            size,
            layer,
            self.logical_topologies["AllGather"],  # 逻辑拓扑：AllGather对应项
            self.all_gather_implementation_per_dimension,  # 全收集实现列表
            involved_dimensions,
            ComType.All_Gather,  # 通信类型：全收集
            pref_scheduling,
            event,
            layer_ptr
        )

    def generate_reduce_scatter(
        self,
        size: int,
        involved_dimensions: list[bool],
        pref_scheduling: SchedulingPolicy,
        layer: int,
        event: Common.EventType,
        layer_ptr: Callable
    ) -> DataSet:
        """生成归约分散操作的数据集（调用通用集合通信生成函数）
        
        Args:
            size: 数据总大小（无符号64位整数）
            involved_dimensions: 各维度是否参与的布尔列表（索引对应维度）
            pref_scheduling: 首选调度策略（如FIFO/LIFO等）
            layer: 层次索引（用于分层调度）
            event: 事件类型（触发后续操作的事件标识）
            layer_ptr: 层次指针（关联的可调用对象，用于事件回调）
            
        Returns:
            DataSet: 生成的归约分散操作数据集对象
        """
        return self.generate_collective(
            size,
            layer,
            self.logical_topologies["ReduceScatter"],  # 逻辑拓扑：归约分散对应项
            self.reduce_scatter_implementation_per_dimension,  # 归约分散实现列表
            involved_dimensions,
            ComType.Reduce_Scatter,  # 通信类型：归约分散
            pref_scheduling,
            event,
            layer_ptr
        )

    def generate_collective(
        self,
        size: int,
        layer_num: int,
        topology: LogicalTopology,
        implementation_per_dimension: List[CollectiveImplementation],
        dimensions_involved: List[bool],
        collective_type: Common.ComType,
        pref_scheduling: SchedulingPolicy,
        event: Common.EventType,
        layer_ptr: PyCallable
    ) -> DataSet:
        chunk_size = self.determine_chunk_size(size, collective_type)
        if self.id == 0:
            print(f"chunk size is: {chunk_size}, size is: {size}, layer_num is: {layer_num}, node: {self.id}")
        recommended_chunk_size = chunk_size
        streams = math.ceil(size / chunk_size)
        dataset = DataSet(streams)
        
        # PHY_MTP条件编译处理（假设启用）
        if event != EventType.NONE and layer_ptr is not None:
            dataset.set_notifier(layer_ptr, event)
        
        pri = self.get_priority(pref_scheduling)
        count = 0
        stream_counter = 0

        if self.id == 0 and (
            self.inter_dimension_scheduling in {InterDimensionScheduling.OfflineGreedy, InterDimensionScheduling.OfflineGreedyFlex}
        ):
            if self.last_scheduled_collective != self.boostedTick():
                self.offline_greedy.reset_loads()
                self.last_scheduled_collective = self.boostedTick()

        while size > 0:
            count += 1
            chunk_size = min(chunk_size, size)
            dim_mapper = list(range(topology.get_num_of_dimensions()))
            
            if collective_type == ComType.All_Gather:
                dim_mapper.reverse()
            
            if self.inter_dimension_scheduling == InterDimensionScheduling.RoundRobin:
                rotate_idx = self.round_robin_inter_dimension_scheduler
                dim_mapper = dim_mapper[rotate_idx:] + dim_mapper[:rotate_idx]
                self.round_robin_inter_dimension_scheduler = (self.round_robin_inter_dimension_scheduler + 1) % topology.get_num_of_dimensions()
            elif collective_type != ComType.All_to_All and (
                self.inter_dimension_scheduling in {InterDimensionScheduling.OfflineGreedy, InterDimensionScheduling.OfflineGreedyFlex}
            ):
                prev_size = size
                dim_mapper = self.offline_greedy.get_chunk_scheduling(
                    stream_counter, size, recommended_chunk_size, dimensions_involved,
                    self.inter_dimension_scheduling, collective_type
                )
                chunk_size = prev_size - size
                size -= chunk_size  # 手动更新size，因为OfflineGreedy可能直接修改了size

            if collective_type == ComType.All_to_All or (
                self.inter_dimension_scheduling not in {InterDimensionScheduling.OfflineGreedy, InterDimensionScheduling.OfflineGreedyFlex}
            ):
                size -= chunk_size

            tmp = chunk_size
            vect: List[CollectivePhase] = []
            phase: CollectivePhase

            if collective_type != ComType.All_Reduce or self.collectiveOptimization == CollectiveOptimization.Baseline:
                for dim in dim_mapper:
                    if (topology.get_num_of_nodes_in_dimension(dim) == 1 or 
                        not dimensions_involved[dim]):
                        continue
                    queue = self.vLevels.get_next_queue_at_level(dim)  # 假设vLevels是类成员
                    phase = self.generate_collective_phase(
                        collective_type, layer_num,
                        topology.get_basic_topology_at_dimension(dim, collective_type),
                        tmp, queue[0], queue[1], InjectionPolicy.Normal,
                        implementation_per_dimension[dim], self.boost_mode
                    )
                    vect.append(phase)
                    tmp = phase.final_data_size
            else:
                # 处理AllReduce的特殊逻辑
                dim = 0
                last_active_dim = 0
                for d in range(topology.get_num_of_dimensions()):
                    if (topology.get_num_of_nodes_in_dimension(dim_mapper[d]) != 1 and 
                        dimensions_involved[dim_mapper[d]]):
                        last_active_dim = d
                # 前向Reduce-Scatter
                for d in range(last_active_dim):
                    dim = dim_mapper[d]
                    if (topology.get_num_of_nodes_in_dimension(dim) == 1 or 
                        not dimensions_involved[dim]):
                        continue
                    queue = self.vLevels.get_next_queue_at_level(dim)
                    phase = self.generate_collective_phase(
                        ComType.Reduce_Scatter, layer_num,
                        topology.get_basic_topology_at_dimension(dim, ComType.Reduce_Scatter),
                        tmp, queue[0], queue[1], InjectionPolicy.Normal,
                        implementation_per_dimension[dim], self.boost_mode
                    )
                    vect.append(phase)
                    tmp = phase.final_data_size
                # 中间All-Reduce
                dim = last_active_dim
                while dim >= 0 and (
                    not dimensions_involved[dim_mapper[dim]] or 
                    topology.get_num_of_nodes_in_dimension(dim_mapper[dim]) == 1
                ):
                    dim -= 1
                if dim >= 0:
                    queue = self.vLevels.get_next_queue_at_level(dim_mapper[dim])
                    phase = self.generate_collective_phase(
                        ComType.All_Reduce, layer_num,
                        topology.get_basic_topology_at_dimension(dim_mapper[dim], ComType.All_Reduce),
                        tmp, queue[0], queue[1], InjectionPolicy.Normal,
                        implementation_per_dimension[dim_mapper[dim]], self.boost_mode
                    )
                    vect.append(phase)
                    tmp = phase.final_data_size
                # 后向All-Gather
                for d in reversed(range(last_active_dim)):
                    dim = dim_mapper[d]
                    if (topology.get_num_of_nodes_in_dimension(dim) == 1 or 
                        not dimensions_involved[dim]):
                        continue
                    queue = self.vLevels.get_next_queue_at_level(dim)
                    phase = self.generate_collective_phase(
                        ComType.All_Gather, layer_num,
                        topology.get_basic_topology_at_dimension(dim, ComType.All_Gather),
                        tmp, queue[0], queue[1], InjectionPolicy.Normal,
                        implementation_per_dimension[dim], self.boost_mode
                    )
                    vect.append(phase)
                    tmp = phase.final_data_size

            if len(vect) > 0:
                newStream = StreamBaseline(self, dataset, stream_counter, vect, pri)
                newStream.current_queue_id = -1
                self.insert_into_ready_list(newStream)
                MockNcclLog.getInstance().writeLog(MockNcclLog.NcclLogLevel.DEBUG, "Sys::generate_collective finished")
                stream_counter += 1
            else:
                dataset.active = False
                break

        if dataset.active:
            self.streams_injected += count
            dataset.total_streams = count

        return dataset

    def generate_collective_phase(
        self,
        collective_type: Common.ComType,
        layer_num: int,
        topology: BasicLogicalTopology,
        data_size: int,
        queue_id: int,
        direction: RingTopology.Direction,
        injection_policy: InjectionPolicy,
        collective_implementation: CollectiveImplementation,
        boost_mode: bool
    ) -> CollectivePhase:
    
        NcclLog = MockNcclLog.getInstance()

        if collective_implementation.type in [CollectiveImplementationType.Ring, CollectiveImplementationType.OneRing]:
            vn = CollectivePhase(
                self,
                queue_id,
                Ring(
                    collective_type,
                    self.id,
                    layer_num,
                    topology,
                    data_size,
                    direction,
                    injection_policy,
                    boost_mode
                )
            )
            return vn
        elif collective_implementation.type in [CollectiveImplementationType.Direct, CollectiveImplementationType.OneDirect]:
            vn = CollectivePhase(
                self,
                queue_id,
                AllToAll(
                    collective_type,
                    collective_implementation.direct_collective_window,
                    self.id,
                    layer_num,
                    topology,
                    data_size,
                    direction,
                    InjectionPolicy.Normal,
                    boost_mode
                )
            )
            return vn
        elif collective_implementation.type == CollectiveImplementationType.DoubleBinaryTree:
            vn = CollectivePhase(
                self,
                queue_id,
                DoubleBinaryTreeAllReduce(
                    self.id, layer_num, topology, data_size, boost_mode
                )
            )
            return vn
        elif collective_implementation.type in [CollectiveImplementationType.HalvingDoubling, CollectiveImplementationType.OneHalvingDoubling]:
            vn = CollectivePhase(
                self,
                queue_id,
                HalvingDoubling(
                    collective_type,
                    self.id,
                    layer_num,
                    topology,
                    data_size,
                    boost_mode
                )
            )
            return vn
        elif collective_implementation.type == CollectiveImplementationType.NcclFlowModel:
            comm_ps = None
            if self.workload.current_state == LoopState.Forward_Pass:
                comm_ps = self.workload.layers[self.workload.index].fwd_pass_group_type
            elif self.workload.current_state == LoopState.Input_Gradient:
                comm_ps = self.workload.layers[self.workload.index].input_grad_group_type
            elif self.workload.current_state == LoopState.Weight_Gradient:
                comm_ps = self.workload.layers[self.workload.index].weight_grad_group_type

            cs = sysCriticalSection()
            nccl_info = self.get_nccl_Info(comm_ps, data_size, collective_type)
            ptr_FlowModels = self.generate_flow_model(comm_ps, data_size, collective_type)
            cs.ExitSection()

            if nccl_info.algorithm == NCCL_ALGO_RING:
                RingFlowModels = ptr_FlowModels

                cs = sysCriticalSection()
                channels = self.mock_nccl_comms[comm_ps].get_rings()
                cs.ExitSection()

                NcclLog.writeLog(NcclLogLevel.DEBUG, f"rank {self.id} generate FlowModels")
                if RingFlowModels is not None:
                    NcclLog.writeLog(NcclLogLevel.DEBUG, f"rank {self.id} NcclMock generate {len(channels)} channel and flow model count: {len(RingFlowModels)}")
                    for flow in RingFlowModels:
                        prev = -1 if len(flow.second.prev) == 0 else flow.second.prev[0]
                        parent_flow_id = -1 if len(flow.second.parent_flow_id) == 0 else flow.second.parent_flow_id[0]
                        child_flow_id = -1 if len(flow.second.child_flow_id) == 0 else flow.second.child_flow_id[0]

                        NcclLog.writeLog(NcclLogLevel.DEBUG, f" {flow.first.first}, {flow.first.second}, {flow.second.src} to {flow.second.dest} current_flow_id {flow.second.flow_id} prev rank: {prev} parent_flow_id: {parent_flow_id} child_flow_id: {child_flow_id} chunk_id: {flow.second.chunk_id} flow_size: {flow.second.flow_size} chunk_count: {flow.second.chunk_count} ")

                vn = CollectivePhase(
                    self,
                    queue_id,
                    NcclTreeFlowModel(
                        collective_type,
                        self.id,
                        layer_num,
                        topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        RingFlowModels,
                        len(channels)
                    )
                )
                return vn
            elif nccl_info.algorithm == NCCL_ALGO_TREE:

                cs = sysCriticalSection()
                TreeFlowModels = ptr_FlowModels
                treechannels = self.mock_nccl_comms[comm_ps].get_treechannels()
                cs.ExitSection()

                vn = CollectivePhase(
                    self,
                    queue_id,
                    NcclTreeFlowModel(
                        collective_type,
                        self.id,
                        layer_num,
                        topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        TreeFlowModels,
                        len(treechannels)
                    )
                )
                return vn
            elif nccl_info.algorithm == NCCL_ALGO_NVLS:
                collective_type = ComType.All_Reduce_NVLS
                RingFlowModels = ptr_FlowModels

                cs = sysCriticalSection()
                treechannels = self.mock_nccl_comms[comm_ps].get_treechannels()
                cs.ExitSection()

                NcclLog.writeLog(NcclLogLevel.DEBUG, f"rank {self.id} generate FlowModels")
                if RingFlowModels is not None:
                    NcclLog.writeLog(NcclLogLevel.DEBUG, f"rank {self.id} NcclMock generate {len(treechannels)} channel and flow model count: {len(RingFlowModels)}")
                    for flow in RingFlowModels:
                        prev = -1 if len(flow.second.prev) == 0 else flow.second.prev[0]
                        parent_flow_id = -1 if len(flow.second.parent_flow_id) == 0 else flow.second.parent_flow_id[0]
                        child_flow_id = -1 if len(flow.second.child_flow_id) == 0 else flow.second.child_flow_id[0]

                        NcclLog.writeLog(NcclLogLevel.DEBUG, f" {flow.first.first}, {flow.first.second}, {flow.second.src} to {flow.second.dest} current_flow_id {flow.second.flow_id} prev rank: {prev} parent_flow_id: {parent_flow_id} child_flow_id: {child_flow_id} chunk_id: {flow.second.chunk_id} flow_size: {flow.second.flow_size} chunk_count: {flow.second.chunk_count} ")

                vn = CollectivePhase(
                    self,
                    queue_id,
                    NcclTreeFlowModel(
                        collective_type,
                        self.id,
                        layer_num,
                        topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        RingFlowModels,
                        len(treechannels)
                    )
                )
                return vn
        else:
            print("Error: No known collective implementation for collective phase")
            sys.exit(1)

    def insert_stream(self, queue: List[BaseStream], base_stream: BaseStream) -> None:
        it = 0  
        queue_len = len(queue)

        if (self.intra_dimension_scheduling == IntraDimensionScheduling.FIFO or
            base_stream.current_queue_id < 0 or
            base_stream.current_com_type in {ComType.All_to_All, ComType.All_Reduce}):
            while it < queue_len:
                stream = queue[it]
                if stream.initialized:
                    it += 1 
                    continue
                elif stream.priority >= base_stream.priority:
                    it += 1
                    continue
                else:
                    break  

        elif self.intra_dimension_scheduling == IntraDimensionScheduling.RG:
            one_to_last = ComType.NONE
            last = ComType.NONE
            while it < queue_len:
                stream = queue[it]
                one_to_last = last
                last = stream.current_com_type

                if stream.initialized:
                    it += 1
                    if it < queue_len and not queue[it].initialized:
                        one_to_last = last
                        last = queue[it].current_com_type
                        it += 1
                    continue
                elif stream.priority > base_stream.priority:
                    it += 1
                    continue
                elif (last == ComType.Reduce_Scatter and one_to_last == ComType.All_Gather) or \
                    (last == ComType.All_Gather and one_to_last == ComType.Reduce_Scatter):
                    it += 1
                    continue
                else:
                    break

        elif self.intra_dimension_scheduling == IntraDimensionScheduling.SmallestFirst:
            while it < queue_len:
                stream = queue[it]
                if stream.initialized:
                    it += 1
                    continue
                elif stream.my_current_phase.initial_data_size < base_stream.my_current_phase.initial_data_size:
                    it += 1
                    continue
                else:
                    break

        elif self.intra_dimension_scheduling == IntraDimensionScheduling.LessRemainingPhaseFirst:
            while it < queue_len:
                stream = queue[it]
                if stream.initialized:
                    it += 1
                    continue
                elif len(stream.phases_to_go) < len(base_stream.phases_to_go):
                    it += 1
                    continue
                else:
                    break

        queue.insert(it, base_stream)

    def proceed_to_next_vnet_baseline(self, stream: StreamBaseline) -> None:
        nccl_log = MockNcclLog.getInstance()
        
        nccl_log.writeLog(
            MockNcclLog.NcclLogLevel.DEBUG,
            f"proceed_to_next_vnet_baseline :: phase1, stream->current_queue_id {stream.current_queue_id}, stream->phases_to_go.size {len(stream.phases_to_go)}"
        )
        previous_vnet = stream.current_queue_id

        if stream.steps_finished == 1:
            self.first_phase_streams -= 1

        if stream.steps_finished != 0 and stream.net_message_counter != 0:
            stream.net_message_latency[-1] /= stream.net_message_counter

        if stream.my_current_phase.algorithm is not None:
            stream.my_current_phase.algorithm = None

        if len(stream.phases_to_go) == 0:
            stream.take_bus_stats_average()  
            stream.dataset.notify_stream_finished(stream)  

        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: phase2")
        if stream.current_queue_id >= 0 and stream.my_current_phase.enabled:
            try:
                target_queue = self.active_Streams[stream.my_current_phase.queue_id]
                for idx, s in enumerate(target_queue):
                    if isinstance(s, StreamBaseline) and s.stream_num == stream.stream_num:
                        del target_queue[idx]
                        break
            except KeyError:
                nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "Target queue not found in active_Streams")

        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: phase2-1")
        if len(stream.phases_to_go) == 0:
            self.total_running_streams -= 1
            if previous_vnet >= 0:
                nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: phase2-1")
                latency = self.boosted_tick() - stream.last_init
                self.scheduler_unit.notify_stream_removed(previous_vnet, latency)
            
            try:
                self.running_list.pop(0)
            except IndexError:
                nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "running_list is empty")
            
            nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: delete stream")
            return  

        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: phase3")
        stream.steps_finished += 1
        next_phase = stream.phases_to_go.popleft() 

        # 更新流状态
        stream.current_queue_id = next_phase.queue_id
        stream.current_com_type = next_phase.comm_type
        stream.my_current_phase = next_phase
        stream.test = 0
        stream.test2 = 0
        stream.initialized = False
        stream.last_phase_change = self.boosted_tick()
        stream.total_packets_sent = 0
        stream.net_message_latency.append(0.0)
        stream.net_message_counter = 0

        nccl_log.writeLog(
            MockNcclLog.NcclLogLevel.DEBUG,
            f"proceed_to_next_vnet_baseline :: phase1, stream->current_queue_id {stream.current_queue_id}, stream->phases_to_go.size {len(stream.phases_to_go)}"
        )

        if stream.my_current_phase.enabled:
            target_queue = self.active_Streams.setdefault(stream.current_queue_id, [])
            self.insert_stream(target_queue, stream)

        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: phase4")
        stream.state = StreamState.Ready

        if previous_vnet >= 0:
            latency = self.boosted_tick() - stream.last_init
            self.scheduler_unit.notify_stream_removed(previous_vnet, latency)

        if hasattr(self, 'ready_list'):
            try:
                self.ready_list.popleft()
                self.first_phase_streams += 1
                self.total_running_streams += 1
            except IndexError:
                nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "ready_list is empty")

        self.scheduler_unit.notify_stream_added(stream.current_queue_id)
        nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "proceed_to_next_vnet_baseline :: exit")


    def determine_chunk_size(self, size: int, type: Common.ComType) -> int:
        chunk_size = size // self.preferred_dataset_splits
        return chunk_size

    def get_priority(self, pref_scheduling: SchedulingPolicy) -> int:
        if pref_scheduling == SchedulingPolicy.NONE:
            if self.scheduling_policy == SchedulingPolicy.LIFO:
                current = Sys.priority_counter
                Sys.priority_counter += 1  # 模拟C++的priority_counter++
                return current
            else:
                current = Sys.priority_counter
                Sys.priority_counter -= 1  # 模拟C++的priority_counter--
                return current
        elif pref_scheduling == SchedulingPolicy.HIGHEST:
            return 100000000  # 固定返回最高优先级值
        else:
            # 与pref_scheduling为NONE时的逻辑一致
            if self.scheduling_policy == SchedulingPolicy.LIFO:
                current = Sys.priority_counter
                Sys.priority_counter += 1
                return current
            else:
                current = Sys.priority_counter
                Sys.priority_counter -= 1
                return current

    @staticmethod
    def handleEvent(self, arg: Optional[BasicEventHandlerData]) -> None:
        if arg is None:
            return

        ehd = arg
        node = ehd.node
        event = ehd.event
        cs = None 
        nccl_log = MockNcclLog.getInstance()

        if event == EventType.CallEvents:
            nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG, "Sys::handleEvent EventType::CallEvents")
            node.iterate()
            del ehd 

        elif event == EventType.RendezvousSend:
            rsd = RendezvousSendData(ehd)
            rsd.send(EventType.General, None)
            del rsd  

        elif event == EventType.RendezvousRecv:
            rrd = RendezvousRecvData(ehd)
            rrd.recv(EventType.General, None) 
            del rrd 

        elif event == EventType.PacketReceived:
            rcehd = RecvPacketEventHadndlerData(ehd)
            owner = rcehd.owner
            owner.consume(rcehd)  
            del rcehd  

        elif event == EventType.PacketSent:
            sendhd = SendPacketEventHandlerData(ehd)
            nccl_log.writeLog(MockNcclLog.NcclLogLevel.DEBUG,
                              f"packet sent, sender id: {sendhd.senderNodeId}, node id: {node.id}")

            # 临界区逻辑（仅当NS3_MTP或PHY_MTP启用时加锁）
            if self.NS3_MTP or self.PHY_MTP:
                cs = sysCriticalSection()

            try:
                if self.all_generators[sendhd.senderNodeId] is None:
                    return 

                key = (sendhd.receiverNodeId, sendhd.tag)
                if key not in self.pending_sends or len(self.pending_sends[key]) == 0:
                    self.is_there_pending_sends[key] = False

                    if (self.finished_workloads == 1 and len(self.event_queue) == 0 and len(self.pending_sends) == 0) or \
                            not self.initialized:
                        del self 
                else:
                    simSendCaller = self.pending_sends[key].pop(0)  
                    if len(self.pending_sends[key]) == 0:
                        self.pending_sends.pop(key, None)  

                    simSendCaller(EventType.General, None) 

            finally:
                if self.NS3_MTP or self.PHY_MTP:
                    cs.ExitSection()

            del sendhd

        elif event == EventType.PacketSentFinshed:
            ehd = SendPacketEventHandlerData(arg)
            if ehd.owner is not None:
                ehd.owner.sendcallback(ehd)

    def generate_time(self, cycles):
        tmp = self.NI.sim_get_time()
        addition = cycles * self.CLOCK_PERIOD
        tmp.time_val = addition
        return tmp

    def generate_net_test_flow_model(self, data_size: int, nums: int) -> dict:

        result: Dict[(int, int), MockNccl.SingleFlow] = {}
        for i in range(nums):
            tmp = MockNccl.SingleFlow()
            tmp.flow_id = i
            tmp.src = 0
            tmp.dest = 1
            tmp.flow_size = data_size
            tmp.parent_flow_id = []
            tmp.child_flow_id = []
            tmp.channel_id = 0
            result[(0, i)] = tmp

        return result

    def generate_nvl_test_flow_model(
        self, data_size: int, nums: int
    ) -> Dict[Tuple[int, int], SingleFlow]:
        result: Dict[Tuple[int, int], SingleFlow] = {}
        for i in range(nums):
            tmp = SingleFlow()  
            tmp.flow_id = i
            tmp.src = 0
            tmp.dest = 1
            tmp.flow_size = data_size
            tmp.parent_flow_id = []  
            tmp.child_flow_id = []   # 同上
            tmp.channel_id = 0
            # C++的make_pair(0, i)对应Python元组(0, i)作为字典键
            result[(0, i)] = tmp
        return result

    def generate_flow_model(
        self, comm_ps: ParallelStrategy, data_size: int, collective_type: Common.ComType
    ) -> Any: 
        """生成NCCL流模型（根据当前工作负载状态获取对应流模型）
        
        Args:
            comm_ps: 并行策略（枚举值）
            data_size: 数据大小
            collective_type: 集合通信类型（如AllReduce、AllGather等）
            
        Returns:
            MockNccl流模型对象（具体类型由get_flow_model决定）
        """
        # 获取对应通信组的MockNcclComm实例
        p_comm: MockNcclComm = self.mock_nccl_comms[comm_ps]
        
        # 转换工作负载状态到MockNccl的State枚举
        current_workload_state = self.workload.current_state
        if current_workload_state == LoopState.Forward_Pass:
            current_state = LoopState.Forward_Pass
        elif current_workload_state == LoopState.Input_Gradient:
            current_state = LoopState.Input_Gradient
        elif current_workload_state == LoopState.Weight_Gradient:
            current_state = LoopState.Weight_Gradient
        else:
            raise ValueError(f"Unsupported workload state: {current_workload_state}")
        
        # 调用通信对象的方法生成流模型
        return p_comm.get_flow_model(
            data_size,
            collective_type,
            self.workload.index,
            current_state
        )

    def get_nccl_Info(
        self, comm_ps: ParallelStrategy, data_size: int, collective_type: Common.ComType
    ) -> ncclInfo:
        comm_obj = self.mock_nccl_comms[comm_ps]
        return comm_obj.get_algo_proto_info(data_size, collective_type)

    def mock_nccl_comms_init(self) -> bool:
        """初始化不同通信组的MockNcclComm实例（基于并行策略）"""
        TP_size = self.total_nodes if self.workload.model_parallel_npu_group == 0 else self.workload.model_parallel_npu_group
        PP_size = 1
        DP_size = self.total_nodes // (TP_size * PP_size)  # 整数除法
        EP_size = self.workload.expert_parallel_npu_group
        DP_EP_size = DP_size // EP_size  # 整数除法

        # 初始化通信组字典（假设mock_nccl_comms是类的成员字典）
        if not hasattr(self, 'mock_nccl_comms'):
            self.mock_nccl_comms = {}

        # 创建并注册不同类型的通信对象
        if TP_size > 1:
            p_comm = MockNcclComm(
                self.id, 
                MockNccl.GroupType.TP,  # 假设GroupType是MockNccl中的枚举
                self.GlobalGroup
            )
            self.mock_nccl_comms[MockNccl.GroupType.TP] = p_comm  # 键使用枚举值

        if DP_size > 1:
            p_comm = MockNcclComm(
                self.id, 
                MockNccl.GroupType.DP, 
                self.GlobalGroup
            )
            self.mock_nccl_comms[MockNccl.GroupType.DP] = p_comm

        if EP_size > 1:
            p_comm = MockNcclComm(
                self.id, 
                MockNccl.GroupType.EP, 
                self.GlobalGroup
            )
            self.mock_nccl_comms[MockNccl.GroupType.EP] = p_comm

        if DP_EP_size > 1:
            p_comm = MockNcclComm(
                self.id, 
                MockNccl.GroupType.DP_EP, 
                self.GlobalGroup
            )
            self.mock_nccl_comms[MockNccl.GroupType.DP_EP] = p_comm

        return True

    def mock_nccl_global_group_init(self) -> bool:
        """初始化全局NCCL组（若未初始化过）"""
        if self.GlobalGroup is not None:
            return True
        
        total_nodes = self.total_nodes
        TP_size = total_nodes if self.workload.model_parallel_npu_group == 0 else self.workload.model_parallel_npu_group
        PP_size = 1
        DP_size = self.all_gpus[0] // (TP_size * PP_size)  # 整数除法
        EP_size = self.workload.expert_parallel_npu_group
        DP_EP_size = DP_size // EP_size  # 整数除法
        
        self.GlobalGroup = MockNcclGroup(
            self.all_gpus[0],
            self.ngpus_per_node,
            TP_size,
            DP_size,
            PP_size,
            EP_size,
            DP_EP_size,
            self.NVSwitchs,
            self.gpu_type
        )
        return True