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
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_covsol_f'], -1) 
        mod = _temp._covsol_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_covsol_d'], -1) 
        mod = _temp._covsol_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def covsol(a, b):
    """..."""

    m = _import_module(v.dtype)
    x = matrix(b.dtype, b.rows(), b.cols())
    m.covsol(a.block, b.block, x.block)
    return x
