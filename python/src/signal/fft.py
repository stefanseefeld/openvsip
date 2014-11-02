#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import vector, matrix
from . import fwd, inv

def _import_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip.signal', globals(), locals(), ['_fft_f'], -1) 
        mod = _temp._fft_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.signal', globals(), locals(), ['_fft_d'], -1) 
        mod = _temp._fft_d
    elif dtype in (complex):
        _temp = __import__('vsip.signal', globals(), locals(), ['_fft_cd'], -1) 
        mod = _temp._fft_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

class fft:

    def __init__(self, dtype, direction, len, scale, n, hint):

        self.dtype = dtype         #: data type
        self.direction = direction #: direction: either `fwd` or `inv`
        self.length = len          #: the length of the FFT
        self.scale = scale         #: scale

        m = _import_module(dtype)
        if direction == fwd:
            self._impl = m.fft(len, scale, n, hint)
        else:
            self._impl = m.ifft(len, scale, n, hint)

        self.input_size = self._impl.input_size   #: the size of the input vector
        self.output_size = self._impl.output_size #: the size of the output vector

    def __call__(self, input, output=None):

        if output:
            self._impl(input.block, output.block)
        else:
            self._impl(input.block)

