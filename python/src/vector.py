#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import _block

class vector:

    def __init__(self, block=None, array=None, dtype=None, length=None):

        if block is not None:
            self.block = block
        else:
            self.block = _block.block(array, dtype, (length,))
        self.dtype = self.block.dtype

    def length(self): return self.block.size(1, 0)
    def __array__(self): return self.block.__array__()

    def real(self): return vector(block=self.block.real())
    def imag(self): return vector(block=self.block.imag())

    def __eq__(self, other):
        if not isinstance(other, vector):
            other = vector(array=other)
        # make sure bool blocks are loaded
        import vsip.bblock
        # if the block types match, dispatch to the appropriate method
        if type(self.block) == type(other.block):
            return vector(block=self.block == other.block)
        # TODO: handle general case
        raise ValueError, "unsupported dtype"

    def __iadd__(self, other):
        if isinstance(other, vector):
            self.block += other.block
        else:
            self.block += other
        return self

    def __isub__(self, other):
        if isinstance(other, vector):
            self.block -= other.block
        else:
            self.block -= other
        return self

    def __imul__(self, other):
        if isinstance(other, vector):
            self.block *= other.block
        else:
            self.block *= other
        return self

    def __idiv__(self, other):
        if isinstance(other, vector):
            self.block /= other.block
        else:
            self.block /= other
        return self

    def __setitem__(self, i, value):
        """Assign to a single value or a subvector, depending on the type of 'i'."""

        if isinstance(value, vector):
            value = value.block
        if isinstance(i, slice):
            _block.subblock(self.block, (i,)).assign(value)
        else:
            self.block.put(i, value)

    def __getitem__(self, i):
        """This operation returns either a single value or a subvector."""

        if isinstance(i, slice):
            return vector(block=_block.subblock(self.block, (i,)))
        else:
            return self.block.get(i)

    def __str__(self):
        _max = 6
        s = 'vector(dtype=%s, shape=(%s) ['%(self.dtype, self.length())
        l = self.length()
        lmax = l > _max and _max or l
        s += ', '.join([str(self.block.get(i)) for i in range(lmax-1)])
        if lmax < l: s += ', ... ])'
        else: s += ', %s])'%self.block.get(lmax-1)
        return s
