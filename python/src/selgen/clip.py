#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
import numpy
from vsip import vector, matrix

def clip(v, lt, ut, lc, uc):

    m = import_module('vsip.selgen.clip', v.dtype)
    if isinstance(v, vector):
        vout = vector(v.dtype, v.length())
    else:
        vout = matrix(v.dtype, v.rows(), v.cols())
    m.clip(v.block, vout.block, lt, ut, lc, uc)
    return vout

def invclip(v, lt, mt, ut, lc, uc):

    m = import_module('vsip.selgen.clip', v.dtype)
    if isinstance(v, vector):
        vout = vector(v.dtype, v.length())
    else:
        vout = matrix(v.dtype, v.rows(), v.cols())
    m.invclip(v.block, vout.block, lt, mt, ut, lc, uc)
    return vout
