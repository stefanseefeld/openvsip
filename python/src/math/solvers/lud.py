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
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_lud_f'], -1) 
        mod = _temp._lud_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math.solvers', globals(), locals(), ['_lud_d'], -1) 
        mod = _temp._lud_d
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

class lud:

    def __init__(self, dtype, length):

        mod = _import_module(dtype)
        self.impl_ = mod.lud(length)

    def decompose(self, m):

        return self.impl_.decompose(m.block)

    def solve(self, op, b):

        x = matrix(b.dtype, b.rows(), b.cols())
        if not self.impl_.solve(b.block, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

