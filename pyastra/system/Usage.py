class Usage:        
  def __init__(self, level: int, start: int, end: int):
    """
      构造函数
      :param level: 层级（整数值）
      :param start: 开始时间（64位无符号整数值）
      :param end: 结束时间（64位无符号整数值）
    """
    self.level = level
    self.start = start
    self.end = end