from vsip.view import FVector
from vsip.cuda import dda
from vsip.cuda.module import Function, Module
import numpy
from struct import pack, unpack

def _pack_arguments(*args):
    """Pack arguments from `args` into a single `struct` and return
    that, together with a list of objects to be destroyed post-call."""

    arg_data = []
    cleanup = []
    format = ""
    for i, arg in enumerate(args):
        if isinstance(arg, int):
            arg_data.append(arg)
            format += 'l'
        elif isinstance(arg, float):
            arg_data.append(arg)
            format += 'd'      
        elif isinstance(arg, numpy.number):
            arg_data.append(arg)
            format += arg.dtype.char
        elif isinstance(arg, FVector):
            data = dda.FVData(arg)
            arg_data.append(data)
            format += "P"
            cleanup.append(data)
        elif isinstance(arg, buffer):
            arg_data.append(arg)
            format += "s"
        else:
            raise TypeError('invalid type on parameter %d (%s)' %(i + 1, type(arg)))

    buf = pack(format, *arg_data)
    return buf, cleanup


def _function_call(func, *args, **kwds):

    grid = kwds.pop("grid", (1,1))
    stream = kwds.pop("stream", None)
    block = kwds.pop("block", None)
    shared = kwds.pop("shared", None)
    texrefs = kwds.pop("texrefs", [])
    time_kernel = kwds.pop("time_kernel", False)

    if kwds:
        raise ValueError(
            "extra keyword arguments: %s" 
            % (",".join(kwds.iterkeys())))

    if block is None:
        raise ValueError, "must specify block size"

    func.set_block_shape(*block)
    buf, cleanup = _pack_arguments(*args)
    func.param_setv(0, buf)
    func.param_set_size(len(buf))

    if shared is not None:
        func.set_shared_size(shared)
        
    for texref in texrefs:
        func.param_set_texref(texref)

    if stream is None:
        if time_kernel:
            Context.synchronize()

            from time import time
            start_time = time()
        func.launch_grid(*grid)
        if time_kernel:
            Context.synchronize()
            
            return time()-start_time
    else:
        assert not time_kernel, "Can't time the kernel on an asynchronous invocation"
        func.launch_grid_async(grid[0], grid[1], stream)


Function.__call__ = _function_call
#Function.prepare = function_prepare
#Function.prepared_call = function_prepared_call
#Function.prepared_timed_call = function_prepared_timed_call
#Function.prepared_async_call = function_prepared_async_call
#Function.__getattr__ = function___getattr__
