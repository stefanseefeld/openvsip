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
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_toepsol_f'], -1) 
        mod = _temp._toepsol_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_toepsol_d'], -1) 
        mod = _temp._toepsol_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def toepsol(a, b, w=None):
    """Solve a real symmetric or complex Hermitian positive definite Toeplitz linear system."""

    m = _import_module(v.dtype)
    x = vector(a.dtype, a.length())
    if not w:
        w = vector(a.dtype, a.length())
    m.covsol(a.block, b.block, w.block, x.block)
    return x
