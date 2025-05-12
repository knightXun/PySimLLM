from collections import deque

class CallTask:
    def __init__(self, time, fun_ptr, fun_arg):
        self.time = time
        self.fun_ptr = fun_ptr
        self.fun_arg = fun_arg

class MockNcclLog:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化日志级别映射（模拟C++枚举）
            cls.NcclLogLevel = type('NcclLogLevel', (), {
                'DEBUG': 'DEBUG',
                'INFO': 'INFO',
                'WARNING': 'WARNING',
                'ERROR': 'ERROR'
            })()
        return cls._instance
    
    def writeLog(self, log_level, format_str, *args):
        print(f"[{log_level}] {format_str % args}")

class PhyNetSim:
    call_list = deque()
    tick = 0
    _nccl_log = MockNcclLog()

    @staticmethod
    def Now() -> float:
        return PhyNetSim.tick

    @staticmethod
    def Run():
        while PhyNetSim.call_list:
            calltask = PhyNetSim.call_list[0]  # 等价于front()
            # 推进时间直到达到任务执行时间
            while PhyNetSim.tick != calltask.time:
                PhyNetSim.tick += 1
            # 移除并执行任务
            PhyNetSim.call_list.popleft()  # 等价于pop()
            
            PhyNetSim._nccl_log.writeLog(
                PhyNetSim._nccl_log.NcclLogLevel.DEBUG,
                "PhyNetSim::Run calltask begin tick %d",
                PhyNetSim.tick
            )
            
            # 执行任务函数
            calltask.fun_ptr(calltask.fun_arg)
            
            PhyNetSim._nccl_log.writeLog(
                PhyNetSim._nccl_log.NcclLogLevel.DEBUG,
                "PhyNetSim::Run calltask end tick %d",
                PhyNetSim.tick
            )

    @staticmethod
    def Schedule(delay: int, fun_ptr, fun_arg):
        time = PhyNetSim.tick + delay
        calltask = CallTask(time, fun_ptr, fun_arg)
        
        PhyNetSim._nccl_log.writeLog(
            PhyNetSim._nccl_log.NcclLogLevel.DEBUG,
            "PhyNetSim::Schedule calltask"
        )
        
        PhyNetSim.call_list.append(calltask)

    @staticmethod
    def Stop():
        pass

    @staticmethod
    def Destory():
        pass