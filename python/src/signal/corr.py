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
        _temp = __import__('vsip.signal', globals(), locals(), ['_corr_f'], -1) 
        mod = _temp._corr_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.signal', globals(), locals(), ['_corr_d'], -1) 
        mod = _temp._corr_d
    elif dtype in (complex):
        _temp = __import__('vsip.signal', globals(), locals(), ['_corr_cd'], -1) 
        mod = _temp._corr_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

class corr:

    def __init__(self, dtype, support, r, i, n, hint):

        self.dtype = dtype
        self.support = support

        m = _import_module(dtype)
        self._impl = m.corr(support, r, i, n, hint)

    def __call__(self, bias, ref, input, output):

        self._impl(bias, ref.block, input.block, output.block)

