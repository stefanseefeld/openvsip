#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
from vsip import vector, matrix

class chold:
    """Symmetric positive definite linear system solver using a Cholesky decomposition."""

    def __init__(self, dtype, uplo, length):

        mod = import_module('vsip.math.solvers.chold', dtype)
        self.uplo = uplo
        self.length = length
        self.impl_ = mod.chold(uplo, length)

    def decompose(self, m):

        return self.impl_.decompose(m.block)

    def solve(self, b):

        x = matrix(b.dtype, b.rows(), b.cols())
        if not self.impl_.solve(b.block, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

