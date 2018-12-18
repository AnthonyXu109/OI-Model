import gc
import datetime
import pynvml

import torch
import numpy as np


class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        frame: a frame to detect current py-file runtime
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, frame, detail=True, path='', verbose=False, device=0):
        self.frame = frame
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + '{:%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'.format(datetime.datetime.now())
        self.verbose = verbose
        self.begin = True
        self.device = device

        self.func_name = frame.f_code.co_name
        self.filename = frame.f_globals["__file__"]
        if (self.filename.endswith(".pyc") or
                self.filename.endswith(".pyo")):
            self.filename = self.filename[:-1]
        self.module_name = self.frame.f_globals["__name__"]
        self.curr_line = self.frame.f_lineno

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def track(self):
        """
        Track the GPU memory usage
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.curr_line = self.frame.f_lineno
        where_str = self.module_name + ' ' + self.func_name + ':' + ' line ' + str(self.curr_line)

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write("GPU Memory Track | {:%d-%b-%y-%H:%M:%S} |".format(datetime.datetime.now())+
                        " Total Used Memory:{:<7.1f}Mb |".format(meminfo.used/1024**2)+
                        " Allocated Memory:{:<7.1f}Mb |".format(torch.cuda.memory_allocated(device=0)/1024**2)+
                        " Max cached Memory:{:<7.1f}Mb |".format(torch.cuda.max_memory_cached(device=0)/1024**2)+
                        " Cached Memory:{:<7.1f}Mb\n\n".format(torch.cuda.memory_cached(device=0)/1024**2))

                self.begin = False

            if self.print_detail is True:
                ts_list = [tensor.size() for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x), tuple(x.size()), ts_list.count(x.size()), np.prod(np.array(x.size()))*4/1024**2)
                                    for x in self.get_tensors()}
                for t, s, n, m in new_tensor_sizes - self.last_tensor_sizes:
                    f.write('+ | '+str(n)+' * Size:{:<20}'.format(str(s))+' | Memory: '+str(m*n)[:6]+' M | {:<20}\n'.format(str(t)))
                for t, s, n, m in self.last_tensor_sizes - new_tensor_sizes:
                    f.write('- | '+str(n)+' * Size:{:<20}'.format(str(s))+' | Memory: '+str(m*n)[:6]+' M | {:<20}\n'.format(str(t)))
                self.last_tensor_sizes = new_tensor_sizes

            f.write("\nAt {:<50}".format(where_str)+
                    "Total Used Memory:{:<7.1f}Mb |".format(meminfo.used/1024**2)+
                    " Allocated Memory:{:<7.1f}Mb |".format(torch.cuda.memory_allocated(device=0)/1024**2)+
                    " Max cached Memory:{:<7.1f}Mb |".format(torch.cuda.max_memory_cached(device=0)/1024**2)+
                    " Cached Memory:{:<7.1f}Mb\n\n".format(torch.cuda.memory_cached(device=0)/1024**2))
            print(" Allocated Memory:{:<7.1f}Mb |".format(torch.cuda.memory_allocated(device=0)/1024**2))

        pynvml.nvmlShutdown()

