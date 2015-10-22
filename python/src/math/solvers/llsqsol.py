#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
from vsip import vector, matrix

def llsqsol(a, b):
    """Solve a linear least squares problem."""

    m = import_module('vsip.math.solvers.llsqsol', a.dtype)
    x = matrix(b.dtype, a.cols(), b.cols())
    m.llsqsol(a.block, b.block, x.block)
    return x
