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
        _temp = __import__('vsip.selgen', globals(), locals(), ['_clip_f'], -1) 
        mod = _temp._clip_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.selgen', globals(), locals(), ['_clip_d'], -1) 
        mod = _temp._clip_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def clip(v, lt, ut, lc, uc):

    m = _import_module(v.dtype)
    if isinstance(v, vector):
        vout = vector(v.dtype, v.length())
    else:
        vout = matrix(v.dtype, v.rows(), v.cols())
    m.clip(v.block, vout.block, lt, ut, lc, uc)
    return vout

def invclip(v, lt, mt, ut, lc, uc):

    m = _import_module(v.dtype)
    if isinstance(v, vector):
        vout = vector(v.dtype, v.length())
    else:
        vout = matrix(v.dtype, v.rows(), v.cols())
    m.invclip(v.block, vout.block, lt, mt, ut, lc, uc)
    return vout
