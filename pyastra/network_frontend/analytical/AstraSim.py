import queue


class CallTask:
    def __init__(self, time, fun_ptr, fun_arg):
        self.time = time
        self.fun_ptr = fun_ptr
        self.fun_arg = fun_arg


class AnaSim:
    call_list = queue.Queue()
    tick = 0

    @staticmethod
    def Now():
        return AnaSim.tick

    @staticmethod
    def Run():
        while not AnaSim.call_list.empty():
            calltask = AnaSim.call_list.get()
            while calltask.time != AnaSim.tick:
                AnaSim.tick += 1
            calltask.fun_ptr(calltask.fun_arg)

    @staticmethod
    def Schedule(delay, fun_ptr, fun_arg):
        time = AnaSim.tick + delay
        calltask = CallTask(time, fun_ptr, fun_arg)
        AnaSim.call_list.put(calltask)

    @staticmethod
    def Stop():
        pass

    @staticmethod
    def Destroy():
        pass
