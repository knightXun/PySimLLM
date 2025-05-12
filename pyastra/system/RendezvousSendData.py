# -*- coding: utf-8 -*-
"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Callable, Optional

# 假设以下依赖类已在其他模块定义（实际使用时需根据项目结构调整导入）
# from .BasicEventHandlerData import BasicEventHandlerData
# from .Common import MetaData, EventType
# from .SimSendCaller import SimSendCaller

class RendezvousSendData(BasicEventHandlerData, MetaData):
    """
    Rendezvous发送数据处理类，继承自BasicEventHandlerData和MetaData
    """
    
    def __init__(
        self,
        node_id: int,
        generator: Any,
        buffer: Any,
        count: int,
        type_: int,  # 避免与Python内置type冲突
        dst: int,
        tag: int,
        request: Any,
        msg_handler: Optional[Callable[[Any], None]] = None,
        fun_arg: Optional[Any] = None
    ) -> None:
        """
        构造函数
        
        Args:
            node_id: 节点ID
            generator: 系统生成器
            buffer: 数据缓冲区
            count: 数据数量
            type_: 数据类型（避免与Python内置type冲突）
            dst: 目标节点
            tag: 消息标签
            request: 模拟请求对象
            msg_handler: 消息处理回调函数（可选）
            fun_arg: 回调函数参数（可选）
        """
        # 调用父类BasicEventHandlerData的构造函数
        super().__init__(generator, EventType.RendezvousSend)
        
        # 初始化SimSendCaller实例（注意参数顺序需与C++版本一致）
        self.send = SimSendCaller(
            generator=generator,
            buffer=buffer,
            count=count,
            type_=type_,
            dst=dst,
            tag=tag,
            request=request,
            msg_handler=msg_handler,
            fun_arg=fun_arg
        )

"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""