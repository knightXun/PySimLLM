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
        log_level_env = os.getenv("AS_LOG_LEVEL")
        if log_level_env and log_level_env.isdigit():
            self._log_level = NcclLogLevel(int(log_level_env))
        else:
            self._log_level = NcclLogLevel.INFO

        if self._log_name:
            self._log_file = open(self._log_name, "a", encoding="utf-8")

    @classmethod
    def set_log_name(cls, log_name: str):
        with cls._lock:
            cls._log_name = f"{cls.LOG_PATH}{log_name}"
            if cls._instance and cls._instance._log_file:
                cls._instance._log_file.close()
            if cls._instance:
                cls._instance._log_file = open(cls._log_name, "a", encoding="utf-8")

    def _get_current_time(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %X")

    def write_log(self, level: NcclLogLevel, format_str: str, *args):
        if level.value < self._log_level.value:
            return

        level_str = level.name

        thread_id = hex(threading.get_ident())

        log_content = format_str.format(*args)

        log_line = (
            f"[{self._get_current_time()}]"
            f"[{level_str}] "
            f"[{thread_id}] "
            f"{log_content}"
        )

        with self._lock:
            if self._log_file:
                self._log_file.write(f"{log_line}\n")
                self._log_file.flush()  

    def close(self):
        with self._lock:
            if self._log_file:
                self._log_file.close()
                self._log_file = None


if __name__ == "__main__":
    MockNcclLog.set_log_name("mock_nccl.log")

    logger = MockNcclLog.get_instance()

    logger.write_log(NcclLogLevel.DEBUG, "调试信息：{}", "变量值")
    logger.write_log(NcclLogLevel.INFO, "正常信息：线程{}启动", "主线程")
    logger.write_log(NcclLogLevel.WARNING, "警告：内存使用率{}%", 85)
    logger.write_log(NcclLogLevel.ERROR, "错误：{}", "文件未找到")

    logger.close()