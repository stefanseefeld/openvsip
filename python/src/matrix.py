#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import _block
from vector import vector

class matrix:

    def __init__(self, block=None, array=None, dtype=None, rows=None, cols=None):

        if block is not None:
            self.block = block
        else:
            self.block = _block.block(array, dtype, (rows,cols))
        self.dtype = self.block.dtype

    def __array__(self): return self.block.__array__()

    def rows(self): return self.block.size(2, 0)
    def cols(self): return self.block.size(2, 1)

    def row(self, r): return vector(block=self.block.row(r))
    def col(self, c): return vector(block=self.block.col(c))
    def submatrix(self, s0, s1):
        return matrix(block=_block.submatrix(self.block, s0, s1))

    def real(self): return matrix(block=self.block.real())
    def imag(self): return matrix(block=self.block.imag())

    def __eq__(self, other):
        if not isinstance(other, matrix):
            other = matrix(array=other)
        # make sure bool blocks are loaded
        import vsip.bblock
        # if the block types match, dispatch to the appropriate method
        if type(self.block) == type(other.block):
            return matrix(block=self.block == other.block)
        # TODO: handle general case
        raise ValueError, "unsupported dtype"

    def __iadd__(self, other):
        if not isinstance(other, vector):
            other = matrix(array=other)
        self.block += other.block
        return self

    def __isub__(self, other):
        if not isinstance(other, vector):
            other = matrix(array=other)
        self.block -= other.block
        return self

    def __imul__(self, other):
        if not isinstance(other, vector):
            other = matrix(array=other)
        self.block *= other.block
        return self

    def __setitem__(self, i, value):
        """Assign to a single value or a subview, depending on the type of 'i'."""

        if isinstance(value, (vector, matrix)):
            value = value.block
        if type(i) != tuple:
            # must be a row accessor
            if isinstance(i, slice):
                _block.subblock(self.block, (i,slice(0,None))).assign(value)
            else:
                self.block.row(i).assign(value)
            return
        assert len(i) == 2
        if isinstance(i[0], slice) and isinstance(i[1], slice):
            _block.subblock(self.block, i).assign(value)
        elif isinstance(i[0], slice):
            _block.subblock(self.block.col(i[1]), (i[0],)).assign(value)
        elif isinstance(i[1], slice):
            _block.subblock(self.block.row(i[0]), (i[1],)).assign(value)
        else:
            self.block.put(i[0], i[1], value)

    def __getitem__(self, i):
        """This operation returns either a single value or a subview."""

        if type(i) != tuple:
            # must be a row accessor
            if isinstance(i, slice):
                return matrix(block=_block.subblock(self.block, (i,slice(0,None))))
            else:
                return vector(block=self.block.row(i))
        assert len(i) == 2
        if isinstance(i[0], slice) and isinstance(i[1], slice):
            return matrix(block=_block.subblock(self.block, i))
        elif isinstance(i[0], slice):
            return vector(block=_block.subblock(self.block.col(i[1]), (i[0],)))
        elif isinstance(i[1], slice):
            return vector(block=_block.subblock(self.block.row(i[0]), (i[1],)))
        else:
            return self.block.get(i[0], i[1])

    def __str__(self):
        _max = 6
        s = 'matrix(dtype=%s, shape=(%s,%s)\n  ['%(self.dtype, self.rows(), self.cols())
        r, c = self.rows(), self.cols()
        rmax = r > _max and _max or r
        cmax = c > _max and _max or c
        for rr in range(rmax-1):
            s += '['
            s += ', '.join([str(self.block.get(rr, i)) for i in range(cmax-1)])
            if cmax < c: s += ', ... ]\n   '
            else: s += ', %s]\n   '%self.block.get(rr, cmax-1)
        if rmax < r: s += '[ ... ]])'
        else:
            s += '['
            s += ', '.join([str(self.block.get(rmax-1, i)) for i in range(cmax-1)])
            if cmax < c: s += ', ... ]])'
            else: s += ', %s]])'%self.block.get(rmax-1, cmax-1)
        return s
