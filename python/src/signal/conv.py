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
        _temp = __import__('vsip.signal', globals(), locals(), ['_conv_f'], -1) 
        mod = _temp._conv_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.signal', globals(), locals(), ['_conv_d'], -1) 
        mod = _temp._conv_d
    elif dtype in (complex):
        _temp = __import__('vsip.signal', globals(), locals(), ['_conv_cd'], -1) 
        mod = _temp._conv_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

class conv:

    def __init__(self, dtype, symmetry, support, i, decimation, n, hint):

        self.dtype = dtype
        self.symmetry = symmetry
        self.support = support

        m = _import_module(dtype)
        self._impl = m.conv(symmetry, i, decimation, support, n, hint)

    def __call__(self, input, output):

        self._impl(input.block, output.block)

