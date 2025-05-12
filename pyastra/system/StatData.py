# AstraSim.py 文件内容

class CallData:
    """假设存在的基类（根据实际情况可能需要用户自定义）"""
    pass

class StatData(CallData):
    """统计数据类实现"""
    def __init__(self):
        self.start = 0
        self.waiting = 0
        self.end = 0