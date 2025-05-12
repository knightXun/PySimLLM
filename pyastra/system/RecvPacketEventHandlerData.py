# 假设BasicEventHandlerData和MetaData类已在当前作用域或可导入的模块中存在
# 假设BaseStream、EventType、ncclFlowTag等类型已定义

class RecvPacketEventHadndlerData(BasicEventHandlerData, MetaData):
    def __init__(self, owner, *args, **kwargs):
        """
        支持两种构造方式：
        1. RecvPacketEventHadndlerData(owner: BaseStream, nodeId: int, event: EventType, vnet: int, stream_num: int)
        2. RecvPacketEventHadndlerData(owner: BaseStream, event: EventType, flowTag: ncclFlowTag)
        """
        # 解析参数类型，判断构造方式
        if len(args) == 4 and all(isinstance(arg, int) for arg in args[:3]) and isinstance(args[3], int):
            # 第一种构造方式参数：nodeId, event, vnet, stream_num
            nodeId, event, vnet, stream_num = args
            # 调用父类BasicEventHandlerData的构造函数（假设owner.owner存在）
            super().__init__(owner.owner, event)
            self.owner = owner
            self.vnet = vnet
            self.stream_num = stream_num
            self.message_end = True
            # 使用Python的time模块获取当前时间戳（假设Tick类型对应Python的int时间单位）
            self.ready_time = int(time.time() * 1e9)  # 纳秒级时间戳模拟
            self.flow_id = -2
            self.channel_id = kwargs.get('channel_id', 0)  # 原C++代码未显式初始化channel_id，这里设默认值
            self.child_flow_id = -1

        elif len(args) == 2 and isinstance(args[0], EventType) and isinstance(args[1], ncclFlowTag):
            # 第二种构造方式参数：event, flowTag
            event, flowTag = args
            super().__init__(owner.owner, event)
            self.flowTag = flowTag

        else:
            raise ValueError("Invalid constructor arguments")

# 说明：
# 1. Python不支持函数重载，通过参数解析实现类似C++的多构造函数效果
# 2. 原C++中的Tick类型用Python的int时间戳模拟（假设是纳秒级）
# 3. channel_id在原C++头文件中声明但源文件未显式初始化，这里设置默认值0
# 4. 需要导入time模块来获取时间戳：import time
# 5. 假设所有前置类型（BaseStream/EventType/ncclFlowTag等）已正确定义
# 6. 父类BasicEventHandlerData的构造函数需要owner.owner参数（与原C++代码保持一致）