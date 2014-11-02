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
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_llsqsol_f'], -1) 
        mod = _temp._llsqsol_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_llsqsol_d'], -1) 
        mod = _temp._llsqsol_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def llsqsol(a, b):
    """..."""

    m = _import_module(a.dtype)
    x = matrix(b.dtype, a.cols(), b.cols())
    m.llsqsol(a.block, b.block, x.block)
    return x
