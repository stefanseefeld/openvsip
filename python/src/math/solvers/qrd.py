#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
from vsip import vector, matrix
from vsip.math.types import mat_op, storage, product_side

class qrd:
    """Over-determined linear system solver using QR decomposition."""

    nosave = storage.qrd_nosaveq
    saveq1 = storage.qrd_saveq1
    saveq = storage.qrd_saveq

    def __init__(self, dtype, rows, cols, storage):

        self._rows = rows
        self._cols = cols
        self.qstorage = storage
        mod = import_module('vsip.math.solvers.qrd', dtype)
        self.impl_ = mod.qrd(rows, cols, storage)

    def decompose(self, m):

        return self.impl_.decompose(m.block)

    def prodq(self, op, side, b):

        if self.qstorage == qrd.saveq1:
            shape = (self._rows, self._cols)
        else:
            shape = (self._rows, self._rows)
        if op in (mat_op.trans, mat_op.herm):
            shape = (shape[1], shape[0])
        if side == product_side.lside:
            shape = (shape[0], b.cols())
        else:
            shape = (b.rows(), shape[1])

        x = matrix(b.dtype, shape[0], shape[1])
        if not self.impl_.prodq(op, side, b.block, x.block):
            raise ValueError, "something went wrong"
        else:
            return x

    def rsol(self, b, alpha):

        x = matrix(b.dtype, b.rows(), b.cols())
        if not self.impl_.rsol(b.block, alpha, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

    def covsol(self, b):

        x = matrix(b.dtype, b.rows(), b.cols())
        if not self.impl_.covsol(b.block, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

    def lsqsol(self, b):

        x = matrix(b.dtype, self._cols, b.cols())
        if not self.impl_.lsqsol(b.block, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

