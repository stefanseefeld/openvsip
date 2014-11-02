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
        _temp = __import__('vsip.signal', globals(), locals(), ['_iir_f'], -1) 
        mod = _temp._iir_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.signal', globals(), locals(), ['_iir_d'], -1) 
        mod = _temp._iir_d
    elif dtype in (complex,):
        _temp = __import__('vsip.signal', globals(), locals(), ['_iir_cd'], -1) 
        mod = _temp._iir_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

fwd = -1
inv = 1

class iir:

    def __init__(self, dtype, a, b, len, state, n, hint):

        self.dtype = dtype

        m = _import_module(self.dtype)
        self._impl = m.iir(a, b, len, state, n, hint)

        #self.input_size = self._impl.input_size
        #self.output_size = self._impl.output_size

    def __call__(self, input, output):

        self._impl(input.block, output.block)

    def reset(self):

        self._impl.reset()
