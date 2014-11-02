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
        _temp = __import__('vsip.signal', globals(), locals(), ['_freqswap_f'], -1) 
        mod = _temp._freqswap_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.signal', globals(), locals(), ['_freqswap_d'], -1) 
        mod = _temp._freqswap_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def freqswap(v):

    m = _import_module(v.dtype)
    if isinstance(v, vector):
        vout = vector(v.dtype, v.length())
    else:
        vout = matrix(v.dtype, v.rows(), v.cols())
    m.freqswap(v.block, vout.block)
    return vout
