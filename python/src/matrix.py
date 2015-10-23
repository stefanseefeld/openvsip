#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import _block
from vsip import import_module
from vector import vector

class matrix:
    """A matrix contains a two-dimensional data-set of a specific type."""

    def __init__(self, *args, **kwargs):
        """Create a matrix. There are different ways to construct a new matrix:
        
           `matrix(dtype, rows, cols)`
              will construct a new uninitialized matrix of type `dtype` and shape `(rows, cols)`.

           `matrix(dtype, rows, cols, value)`
              will construct a new matrix of type `dtype` and shape `(rows, cols)` initialized to value.

           `matrix(array=A)`
              will construct a matrix from an existing sequence. data type and length
              will be inferred from `A`

           `matrix(block=B)`
              will construct a matrix from an existing block `B`. data type and length
              will be inferred from `B`."""

        self.block = None #: the block object
        self.dtype = None #: the data type
        
        if args:

            if len(args) == 3:
                self.block = _block.block(args[0], (args[1],args[2]))
            elif len(args) == 4:
                self.block = _block.block(args[0], (args[1],args[2]), args[3])
            else:
                raise ValueError, 'Invalid arguments'
        elif 'block' in kwargs:
            self.block = kwargs['block']
        elif 'array' in kwargs:
            self.block = _block.block(array=kwargs['array'])
        else:
            raise ValueError, 'Invalid arguments'
        self.dtype = self.block.dtype

    def __array__(self): return self.block.__array__()

    def rows(self):
        """Return the number of rows of the matrix."""

        return self.block.size(2, 0)

    def cols(self):
        """Return the number of columns of the matrix."""

        return self.block.size(2, 1)

    shape = property(lambda self:(self.rows(),self.cols()))

    def size(self, d):
        """Return the size in the dimension specified by `d`."""

        return d == 0 and self.rows() or self.cols()

    def local(self):
        """Return the local submatrix if this is a distributed matrix, 'self' otherwise."""

        import_module('vsip.block', self.block.dtype)
        return matrix(block=self.block.local())
        
    def row(self, r):
        """Retur the `r`'th row."""
        
        return vector(block=self.block.row(r))

    def col(self, c):
        """Return the `c`'th column."""

        return vector(block=self.block.col(c))

    def diag(self):
        """Return a diagonal vector."""

        return vector(block=self.block.diag())

    def submatrix(self, s0, s1):
        """Return a submatrix according to slices `s0` and `s1`.
        This is another spelling of `__getitem__(self, s0, s1)`."""
        
        return matrix(block=_block.submatrix(self.block, s0, s1))

    def real(self):
        """Return the real component of this matrix."""

        return matrix(block=self.block.real())

    def imag(self):
        """Return the imaginary component of this matrix."""

        return matrix(block=self.block.imag())

    def __eq__(self, other):
        """Perform an element-wise comparison to `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 == m2
          matrix(dtype=bool, shape=(2,2)
            [[True, False],
             [True, True]])'
        """


        if not isinstance(other, matrix):
            other = matrix(array=other)
        # make sure bool blocks are loaded
        import vsip._block_b
        # if the block types match, dispatch to the appropriate method
        if type(self.block) == type(other.block):
            return matrix(block=self.block == other.block)
        # TODO: handle general case
        raise ValueError, "unsupported dtype"

    def __iadd__(self, other):
        """Perform an elementwise in-place addition of `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 += m2
          >>> m1
          matrix(dtype=float64, shape=(2,2)
            [[2.0, 2.0],
             [6.0, 8.0]])
        """

        if not isinstance(other, matrix):
            self.block += other
        else:
            self.block += other.block
        return self

    def __isub__(self, other):
        """Perform an elementwise in-place substraction of `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 -= m2
          >>> m1
          matrix(dtype=float64, shape=(2,2)
            [[0.0, 2.0],
             [0.0, 0.0]])
        """

        if not isinstance(other, matrix):
            self.block -= other
        else:
            self.block -= other.block
        return self

    def __imul__(self, other):
        """Perform an elementwise in-place multiplication of `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 *= m2
          >>> m1
          matrix(dtype=float64, shape=(2,2)
            [[1.0, 0.0],
             [9.0, 16.0]])
        """

        if isinstance(other, matrix):
            self.block *= other.block
        else:
            self.block *= other
        return self

    def __idiv__(self, other):
        """Perform an elementwise in-place division of `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 /= m2
          >>> m1
          matrix(dtype=float64, shape=(2,2)
            [[1.0, nan],
             [1.0, 1.0]])
        """

        if isinstance(other, matrix):
            self.block /= other.block
        else:
            self.block /= other
        return self

    def __neg__(self):
        """Perform an elementwise negation of `self`. Example:
        
          >>> m = matrix(array=[[1.,2.],[3.,4.]])
          >>> -m
          matrix(dtype=float64, shape=(2,2)
            [[-1.0, -2.0],
             [-3.0, -4.0]])
        """

        m = matrix(block=-self.block)
        return m

    def __add__(self, other):
        """Perform an elementwise addition of `self` and `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 + m2
          matrix(dtype=float64, shape=(2,2)
            [[2.0, 2.0],
             [6.0, 8.0]])
        """

        m = matrix(block=self.block.copy())
        m+= other
        return m

    def __sub__(self, other):
        """Perform an elementwise substraction of `self` and `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 - m2
          matrix(dtype=float64, shape=(2,2)
            [[0.0, 2.0],
             [0.0, 0.0]])
        """

        m = matrix(block=self.block.copy())
        m-= other
        return m

    def __mul__(self, other):
        """Perform an elementwise multiplication of `self` and `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 * m2
          matrix(dtype=float64, shape=(2,2)
            [[1.0, 0.0],
             [9.0, 16.0]])
        """

        m = matrix(block=self.block.copy())
        m*= other
        return m

    def __rmul__(self, other):

        # Allow right-multiplication by a scalar of known type
        if type(other) in (int, float, complex):
            m = matrix(block=self.block.copy())
            m*= other
            return m
        else:
            return NotImplemented

    def __div__(self, other):
        """Perform an elementwise division of `self` and `other`. Example:
        
          >>> m1 = matrix(array=[[1.,2.],[3.,4.]])
          >>> m2 = matrix(array=[[1.,0.],[3.,4.]])
          >>> m1 / m2
          matrix(dtype=float64, shape=(2,2)
            [[1.0, nan],
             [1.0, 1.0]])
        """

        m = matrix(block=self.block.copy())
        m/= other
        return m

    def __getitem__(self, i):
        """This operation returns either a single value or a subview. Examples:

          >>> m = matrix(array=[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
          >>> m[1]         # row 1
          vector(dtype=float64, shape=(3), data=[4.,5.,6.])
          >>> m[:,1]       # column 1
          vector(dtype=float64, shape=(3), data=[2.,5.,8.])
          >>> m[0,0]       # element (0,0)
          1.0
          >>> m[0:2,0:3:2] # a submatrix
          matrix(dtype=float64, shape=(2,2)
            [[1.0, 3.0],
             [4.0, 6.0]])
        """

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

    def __setitem__(self, i, value):
        """Assign to a single value or a subview. Examples:

          >>> m = matrix(array=[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
          >>> m[1] = 42.         # row 1
          >>> m
          matrix(dtype=float64, shape=(3,3)
            [[1.0, 2.0, 3.0],
             [42.0, 42.0, 42.0],
             [7.0, 8.0, 9.0]])
          >>> m[:,1] = 7.        # column 1
          >>> m
          matrix(dtype=float64, shape=(3,3)
            [[1.0, 7.0, 3.0],
             [42.0, 7.0, 42.0],
             [7.0, 7.0, 9.0]])
          >>> m[0,0] = 12.       # element (0,0)
          matrix(dtype=float64, shape=(3,3)
            [[12.0, 7.0, 3.0],
             [42.0, 7.0, 42.0],
             [7.0, 7.0, 9.0]])
          >>> m[0:2,0:3:2] = 13  # a submatrix
          matrix(dtype=float64, shape=(3,3)
            [[13.0, 7.0, 13.0],
             [13.0, 7.0, 13.0],
             [7.0, 7.0, 9.0]])
        """

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
            if i[0] == slice(None) and i[1] == slice(None):
                self.block.assign(value)
            else:
                _block.subblock(self.block, i).assign(value)
        elif isinstance(i[0], slice):
            _block.subblock(self.block.col(i[1]), (i[0],)).assign(value)
        elif isinstance(i[1], slice):
            _block.subblock(self.block.row(i[0]), (i[1],)).assign(value)
        else:
            self.block.put(i[0], i[1], value)

    def __str__(self):
        _max = 6
        s = '['
        r, c = self.rows(), self.cols()
        rmax = r > _max and _max or r
        cmax = c > _max and _max or c
        for rr in range(rmax-1):
            s += '['
            s += ', '.join([str(self.block.get(rr, i)) for i in range(cmax-1)])
            if c == 1: s += '%s]\n   '%self.block.get(rr, 0)
            elif cmax < c: s += ', ... ]\n   '
            else: s += ', %s]\n   '%self.block.get(rr, cmax-1)
        if rmax < r: s += '[ ... ]]'
        else:
            s += '['
            s += ', '.join([str(self.block.get(rmax-1, i)) for i in range(cmax-1)])
            if c == 1: s += '%s]]\n   '%self.block.get(rmax-1, cmax-1)
            elif cmax < c: s += ', ... ]]'
            else: s += ', %s]]'%self.block.get(rmax-1, cmax-1)
        return s

    def __repr__(self):
        s = 'matrix(dtype=%s, shape=(%s,%s)\n  %s)'%(self.dtype, self.rows(), self.cols(), self)
        return s
