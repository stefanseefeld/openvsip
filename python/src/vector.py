#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import _block

class vector:
    """A vector contains a one-dimensional data-set of a specific type."""

    def __init__(self, *args, **kwargs):
        """Create a vector. There are different ways to construct a new vector:
        
           `vector(dtype, len)`
              will construct a new uninitialized vector of type ``dtype`` and length ``len``.

           `vector(dtype, len, value)`
              will construct a new vector of type ``dtype`` and length ``len`` initialized to ``value``.

           `vector(array=A)`
              will construct a vector from an existing sequence. data type and length
              will be inferred from ``A``

           `vector(block=B)`
              will construct a vector from an existing block ``B``. data type and length
              will be inferred from ``B``."""

        self.block = None #: the block object
        self.dtype = None #: the vector's data type

        if args:

            if len(args) == 2:
                self.block = _block.block(args[0], (args[1],))
            elif len(args) == 3:
                self.block = _block.block(args[0], (args[1],), args[2])
            else:
                raise ValueError, 'Invalid arguments'
        elif 'block' in kwargs:
            self.block = kwargs['block']
        elif 'array' in kwargs:
            self.block = _block.block(array=kwargs['array'])
        else:
            raise ValueError, 'Invalid arguments'
        self.dtype = self.block.dtype

    def __array__(self):
        """Convert this vector to a NumPy array."""

        return self.block.__array__()

    def length(self):
        """Return the length of the vector."""

        return self.block.size(1, 0)

    def size(self):
        """Same as `length`."""

        return self.length()

    def real(self):
        """Return the real component of this vector."""

        return vector(block=self.block.real())

    def imag(self):
        """Return the imaginary component of this vector."""

        return vector(block=self.block.imag())

    def __eq__(self, other):
        """Perform an element-wise comparison to `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 == v2
          vector(dtype=bool, shape=(4) data=[True, False, True, True])
        """

        if not isinstance(other, vector):
            other = vector(array=other)
        # make sure bool blocks are loaded
        import vsip._block_b
        # if the block types match, dispatch to the appropriate method
        if type(self.block) == type(other.block):
            return vector(block=self.block == other.block)
        # TODO: handle general case
        raise ValueError, "unsupported dtype"

    def __iadd__(self, other):
        """Perform an elementwise in-place addition of `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 += v2
          >>> v1
          vector(dtype=float64, shape=(4) data=[2.0, 2.0, 6.0, 8.0])
        """

        if isinstance(other, vector):
            self.block += other.block
        else:
            self.block += other
        return self

    def __isub__(self, other):
        """Perform an elementwise in-place substraction of `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 -= v2
          >>> v1
          vector(dtype=float64, shape=(4) data=[0.0, 2.0, 0.0, 0.0])
        """

        if isinstance(other, vector):
            self.block -= other.block
        else:
            self.block -= other
        return self

    def __imul__(self, other):
        """Perform an elementwise in-place multiplication with `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 *= v2
          >>> v1
          vector(dtype=float64, shape=(4) data=[1.0, 0.0, 9.0, 16.0])
        """

        if isinstance(other, vector):
            self.block *= other.block
        else:
            self.block *= other
        return self

    def __idiv__(self, other):
        """Perform an elementwise in-place division by `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 /= v2
          >>> v1
          vector(dtype=float64, shape=(4) data=[1.0, nan, 1.0, 1.0])
        """

        if isinstance(other, vector):
            self.block /= other.block
        else:
            self.block /= other
        return self

    def __neg__(self):
        """Perform an elementwise negation of `self`. Example:
        
          >>> v = vector(array=[1.,2.,3.,4.])
          >>> -v
          vector(dtype=float64, shape=(4) data=[-1.0, -2.0, -3.0, -4.0])
        """

        v = vector(block=-self.block)
        return v

    def __add__(self, other):
        """Perform an elementwise addition of `self` and `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 + v2
          vector(dtype=float64, shape=(4) data=[2.0, 2.0, 6.0, 8.0])
        """

        v = vector(block=self.block.copy())
        v+= other
        return v

    def __sub__(self, other):
        """Perform an elementwise substraction of `self` and `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 - v2
          vector(dtype=float64, shape=(4) data=[0.0, 2.0, 0.0, 0.0])
        """

        v = vector(block=self.block.copy())
        v-= other
        return v

    def __mul__(self, other):
        """Perform an elementwise multiplication of `self` with `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 * v2
          vector(dtype=float64, shape=(4) data=[1.0, 0.0, 9.0, 16.0])
        """

        v = vector(block=self.block.copy())
        v*= other
        return v

    def __rmul__(self, other):

        # Allow right-multiplication by a scalar of known type
        if type(other) in (int, float, complex):
            v = vector(block=self.block.copy())
            v*= other
            return v
        else:
            return NotImplemented

    def __div__(self, other):
        """Perform an elementwise division of `self` by `other`. Example:
        
          >>> v1 = vector(array=[1.,2.,3.,4.])
          >>> v2 = vector(array=[1.,0.,3.,4.])
          >>> v1 / v2
          vector(dtype=float64, shape=(4) data=[1.0, nan, 1.0, 1.0])
        """

        v = vector(block=self.block.copy())
        v/= other
        return v

    def __getitem__(self, i):
        """This operation returns either a single value or a subvector. Examples:

          >>> v = vector(array=[1.,2.,3.,4.])
          >>> v[1]
          2.
          >>> v[0:4:2]
          vector(dtype=float64, shape=(2) data=[1.0, 3.0])
        """

        if isinstance(i, slice):
            return vector(block=_block.subblock(self.block, (i,)))
        else:
            return self.block.get(i)

    def __setitem__(self, i, value):
        """Assign to a single value or a subvector, depending on the type of 'i'. Examples:

          >>> v = vector(array=[1.,2.,3.,4.])
          >>> v[1] = 42.
          >>> v
          vector(dtype=float64, shape=(4) data=[1.0, 42., 3.0, 4.0])
          >>> v[0:4:2] = 42.
          >>> v
          vector(dtype=float64, shape=(4) data=[42.0, 42.0, 42.0, 4.0])
        """

        if isinstance(value, vector):
            value = value.block
        if isinstance(i, slice):
            _block.subblock(self.block, (i,)).assign(value)
        else:
            self.block.put(i, value)

    def __str__(self):
        """ Return a string representation of `self`"""

        _max = 6
        s = '['
        l = self.length()
        lmax = l > _max and _max or l
        s += ', '.join([str(self.block.get(i)) for i in range(lmax-1)])
        if l == 1: s += ' %s]'%self.block.get(0)
        elif lmax < l: s += ', ... ]'
        else: s += ', %s]'%self.block.get(lmax-1)
        return s

    def __repr__(self):
        s = 'vector(dtype=%s, shape=(%s) data=%s)'%(self.dtype, self.length(), self)
        return s
