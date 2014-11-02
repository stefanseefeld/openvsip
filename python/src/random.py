#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import vector, matrix
from vsip._block import _import_block_module

def _import_module(dtype):

    # Make sure the proper block module is loaded, too
    _import_block_module(dtype)

    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip', globals(), locals(), ['_rand_f'], -1) 
        mod = _temp._rand_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip', globals(), locals(), ['_rand_d'], -1) 
        mod = _temp._rand_d
    elif dtype == complex:
        _temp = __import__('vsip', globals(), locals(), ['_rand_cd'], -1) 
        mod = _temp._rand_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod



class rand:
    """A Random Number Generator"""

    def __init__(self, dtype, seed=0, portable=False):
        """Create a random number generator.

        arguments:

          :dtype: type of the data to be generated
          :seed: initial seed
          :portable: whether or not to use a portable (reproducible) sequence
        """

        mod = _import_module(dtype)
        self._rand = mod.rand(seed, portable)

    def randu(self, *shape):
        """Produce uniformly distributed numbers from the open interval :math:`(0,1)`

        arguments:

          :shape: the shape of the object to be returned:

                    - no argument: return a single number
                    - a single number: return a vector of the given length
                    - two numbers: return a matrix of the given number of rows and columns
        """
        
        if len(shape) == 0:
            return self._rand.randu()
        elif len(shape) == 1:
            return vector(block=self._rand.randu(shape[0]))
        else:
            return matrix(block=self._rand.randu(shape[0], shape[1]))

    def randn(self, *shape):
        """Produce normal-distributed numbers from the open interval :math:`(0,1)`

        arguments:

          :shape: the shape of the object to be returned:

                    - no argument: return a single number
                    - a single number: return a vector of the given length
                    - two numbers: return a matrix of the given number of rows and columns
        """
        
        if len(shape) == 0:
            return self._rand.randn()
        elif len(shape) == 1:
            return vector(block=self._rand.randn(shape[0]))
        else:
            return matrix(block=self._rand.randn(shape[0], shape[1]))
