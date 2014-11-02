#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import vector, matrix

def _import_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip.math', globals(), locals(), ['_reductions_f'], -1) 
        mod = _temp._reductions_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math', globals(), locals(), ['_reductions_d'], -1) 
        mod = _temp._reductions_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def reduce(func, a):

    from vsip import types
    b = a.block
    m = _import_module(b.dtype)
    f = getattr(m, func)
    return f(b)

def meanval(a): return reduce('meanval', a)
def maxval(a): return reduce('maxval', a)
def minval(a): return reduce('minval', a)
def sumval(a): return reduce('sumval', a)
