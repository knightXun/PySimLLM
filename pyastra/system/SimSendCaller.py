# ******************************************************************************
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ******************************************************************************

from typing import Any, Callable as PyCallable

class SimRecvCaller:
    def __init__(
        self,
        generator: Any,  # 对应C++的Sys*类型，实际类型需根据项目定义
        buffer: Any,
        count: int,
        type: int,
        src: int,
        tag: int,
        request: Any,  # 对应sim_request类型，实际类型需根据项目定义
        msg_handler: PyCallable[[Any], None],  # 函数指针转换为Python可调用对象
        fun_arg: Any
    ):
        self.generator = generator
        self.buffer = buffer
        self.count = count
        self.type = type
        self.src = src
        self.tag = tag
        self.request = request
        self.msg_handler = msg_handler
        self.fun_arg = fun_arg

    def call(self, type: int, data: Any) -> None:  # EventType转换为int类型（需根据实际枚举调整）
        """
        核心回调方法，调用生成器的网络接口执行接收操作
        """
        # 注意：Python不需要手动释放对象（自动垃圾回收），因此移除了delete this
        self.generator.NI.sim_recv(
            self.buffer,
            self.count,
            self.type,
            self.src,
            self.tag,
            self.request,  # 注意：C++中使用&this->request，Python直接传递对象引用
            self.msg_handler,
            self.fun_arg
        )