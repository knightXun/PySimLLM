import os
import datetime
import threading
from enum import Enum
from typing import Optional


class NcclLogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class MockNcclLog:
    _instance: Optional['MockNcclLog'] = None
    _log_level: NcclLogLevel = NcclLogLevel.INFO
    _log_name: str = ""
    _lock = threading.Lock()
    _log_file = None

    LOG_PATH = "/etc/astra-sim/"

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._init_log()
            return cls._instance

    def _init_log(self):
        # 从环境变量获取日志级别（默认INFO）
        log_level_env = os.getenv("AS_LOG_LEVEL")
        if log_level_env and log_level_env.isdigit():
            self._log_level = NcclLogLevel(int(log_level_env))
        else:
            self._log_level = NcclLogLevel.INFO

        # 初始化日志文件（如果已设置文件名）
        if self._log_name:
            self._log_file = open(self._log_name, "a", encoding="utf-8")

    @classmethod
    def set_log_name(cls, log_name: str):
        """设置日志文件名（包含路径）"""
        with cls._lock:
            cls._log_name = f"{cls.LOG_PATH}{log_name}"
            # 如果实例已存在，重新打开文件
            if cls._instance and cls._instance._log_file:
                cls._instance._log_file.close()
            if cls._instance:
                cls._instance._log_file = open(cls._log_name, "a", encoding="utf-8")

    def _get_current_time(self) -> str:
        """获取当前时间字符串（格式：YYYY-MM-DD HH:MM:SS）"""
        return datetime.datetime.now().strftime("%Y-%m-%d %X")

    def write_log(self, level: NcclLogLevel, format_str: str, *args):
        """写入日志内容"""
        if level.value < self._log_level.value:
            return

        # 转换日志级别为字符串
        level_str = level.name

        # 获取线程ID（十六进制）
        thread_id = hex(threading.get_ident())

        # 格式化日志内容
        log_content = format_str.format(*args)

        # 构造完整日志行
        log_line = (
            f"[{self._get_current_time()}]"
            f"[{level_str}] "
            f"[{thread_id}] "
            f"{log_content}"
        )

        # 线程安全写入
        with self._lock:
            if self._log_file:
                self._log_file.write(f"{log_line}\n")
                self._log_file.flush()  # 立即刷新确保写入

    def close(self):
        """关闭日志文件（建议程序结束时调用）"""
        with self._lock:
            if self._log_file:
                self._log_file.close()
                self._log_file = None


# 使用示例
if __name__ == "__main__":
    # 设置日志文件名（需先设置，否则不会写入文件）
    MockNcclLog.set_log_name("mock_nccl.log")

    # 获取单例实例
    logger = MockNcclLog()

    # 写入不同级别的日志
    logger.write_log(NcclLogLevel.DEBUG, "调试信息：{}", "变量值")
    logger.write_log(NcclLogLevel.INFO, "正常信息：线程{}启动", "主线程")
    logger.write_log(NcclLogLevel.WARNING, "警告：内存使用率{}%", 85)
    logger.write_log(NcclLogLevel.ERROR, "错误：{}", "文件未找到")

    # 关闭日志文件（可选）
    logger.close()