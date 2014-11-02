#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import vector, matrix
from vsip.signal import *

def _import_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip.signal', globals(), locals(), ['_fftm_f'], -1) 
        mod = _temp._fftm_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.signal', globals(), locals(), ['_fftm_d'], -1) 
        mod = _temp._fftm_d
    elif dtype in (complex,):
        _temp = __import__('vsip.signal', globals(), locals(), ['_fftm_cd'], -1) 
        mod = _temp._fftm_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

class fftm:

    def __init__(self, dtype, direction, rows, cols, scale, axis, n, hint):

        self.dtype = dtype
        self.direction = direction
        self.size = (rows, cols)
        self.scale = scale
        self.axis = axis

        m = _import_module(dtype)
        if direction == fwd:
            self._impl = m.fftm(rows, cols, scale, axis, n, hint)
        else:
            self._impl = m.ifftm(rows, cols, scale, axis, n, hint)

        self.input_size = self._impl.input_size
        self.output_size = self._impl.output_size

    def __call__(self, input, output=None):

        if output:
            self._impl(input.block, output.block)
        else:
            self._impl(input.block)

