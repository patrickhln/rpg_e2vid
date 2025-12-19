import torch
import time
import numpy as np
import atexit

cuda_timers = {}
timers = {}


class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []
            
        try:
            from torch.cuda import Event as CudaEvent
            if torch.cuda.is_available():
                self.cuda_supported = True
            else:
                self.cuda_supported = False
        except Exception:
            self.cuda_supported = False
        
        if not self.cuda_supported:
            class CudaEvent:
                def __init__(self, *args, **kwargs): pass
                def record(self, *args, **kwargs): pass
                def elapsed_time(self, *args, **kwargs): return 0

        self.start = CudaEvent(enable_timing=True) if self.cuda_supported else CudaEvent()
        self.end   = CudaEvent(enable_timing=True) if self.cuda_supported else CudaEvent()

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        if self.cuda_supported:
            import torch
            torch.cuda.synchronize()
            cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))
        else:
            cuda_timers[self.timer_name].append(0)

class Timer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        if self.timer_name not in timers:
            timers[self.timer_name] = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # measured in seconds
        self.interval *= 1000.0  # convert to milliseconds
        timers[self.timer_name].append(self.interval)


def print_timing_info():
    print('== Timing statistics ==')
    for timer_name, timing_values in [*cuda_timers.items(), *timers.items()]:
        timing_value = np.mean(np.array(timing_values))
        if timing_value < 1000.0:
            print('{}: {:.2f} ms'.format(timer_name, timing_value))
        else:
            print('{}: {:.2f} s'.format(timer_name, timing_value / 1000.0))


# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)
