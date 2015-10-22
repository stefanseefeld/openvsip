#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
from vsip import vector, matrix

class lud:
    """General square linear system solver using lower-upper decomposition."""

    def __init__(self, dtype, length):

        m = import_module('vsip.math.solvers.lud', dtype)
        self.impl_ = mod.lud(length)

    def decompose(self, m):

        return self.impl_.decompose(m.block)

    def solve(self, op, b):

        x = matrix(b.dtype, b.rows(), b.cols())
        if not self.impl_.solve(b.block, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

