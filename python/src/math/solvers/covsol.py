#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
from vsip import vector, matrix

def covsol(a, b):
    """Solve a covariance linear system."""

    m = import_module('vsip.math.solvers.covsol', dtype)
    m = import_module(v.dtype)
    x = matrix(b.dtype, b.rows(), b.cols())
    m.covsol(a.block, b.block, x.block)
    return x
