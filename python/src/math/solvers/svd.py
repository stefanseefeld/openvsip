#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import import_module
from vsip import vector, matrix
from vsip.math.types import storage, product_side

class svd:

    uvnos = storage.svd_uvnos
    uvpart = storage.svd_uvpart
    uvfull = storage.svd_uvfull

    def __init__(self, dtype, rows, cols, ustorage, vstorage):

        self.dtype = dtype
        self._rows = rows
        self._cols = cols
        self.ustorage = ustorage
        self.vstorage = vstorage
        mod = import_module('vsip.math.solvers.svd', dtype)
        self.impl_ = mod.svd(rows, cols, ustorage, vstorage)

    def decompose(self, m):

        x = vector(m.dtype, min(m.rows(), m.cols()))
        if not self.impl_.decompose(m.block, x.block):
            raise ValueError, "something went wrong"
        else:
            return x
            

    def produ(self, op, side, b):

        
        shape = self._produv_shape(op, side, b)
        x = matrix(b.dtype, shape[0], shape[1])
        if not self.impl_.produ(op, side, b.block, x.block):
            raise ValueError, "something went wrong"
        else:
            return x

    def prodv(self, op, side, b):

        shape = self._produv_shape(op, side, b)
        x = matrix(b.dtype, shape[0], shape[1])
        if not self.impl_.prodv(op, side, b.block, x.block):
            raise ValueError, "something went wrong"
        else:
            return x

    def u(self, low, high):

        shape = (self._rows, high - low + 1)
        x = matrix(self.dtype, shape[0], shape[1])
        if not self.impl_.u(low, high, x.block):
            raise ValueError, "something went wrong."
        else:
            return x

    def v(self, low, high):

        shape = (self._cols, high - low + 1)
        x = matrix(self.dtype, shape[0], shape[1])
        if not self.impl_.v(low, high, x.block):
            raise ValueError, "something went wrong."
        else:
            return x


    def _select_dim(storage, full, part):

        if storage == uvfull: return full
        elif storage == uvpart: return part
        else: return 0

    def _produv_shape(self, op, side, b):

        uv_shape = (self._select_dim(self.ustorage, self._rows, self._rows),
                    self._select_dim(self.vstorage, self._rows, min(self._rows, self._cols)))
        if side == mat_lside:
            if op == mat_ntrans:
                shape = (uv_shape[0], b.size(1))
            elif op in (mat_trans, mat_herm):
                shape = (uv_shape[1], b.size(1))
        else: # mat_rside
            if op == mat_ntrans:
                shape = (b.size(0), uv_shape[1])
            elif op in (mat_trans, mat_herm):
                shape = (b.size(0), uv_shape[0])
        return shape
