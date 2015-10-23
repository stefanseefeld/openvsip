#
# Copyright (c) 2015 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
import numpy

def block(*args, **kwargs):
    """Create a distributed block.

        Valid argument combinations:

        block(array, map): create a block from an existing array-like container
        block(dtype, shape, map): create a block of the given type and shape
        block(dtype, shape, value, map): create a block of the given type and shape and initialize it to value
    """

    # Unpack constructor arguments
    array = value = None
    if len(args) == 2:
        array, map = args[0], args[1]
        array = numpy.array(array, copy=False)
        shape = array.shape
    elif len(args) == 3:
        dtype, shape, map = args[0], args[1], args[2]
        value = None
    elif len(args) == 4:
        dtype, shape, value, map = args[0], args[1], args[2], args[3]

    mod = import_module('vsip.parallel.block', dtype)

    if 'array' in kwargs:
        return mod.block(array, map)
        
    # Create a new block
    if len(shape) == 1:
        return value and mod.block(shape[0], value, map) or mod.block(shape[0], map)
    elif len(shape) == 2:
        return value and mod.block(shape[0], shape[1], value, map) or mod.block(shape[0], shape[1], map)
    elif len(shape) == 3:
        return value and mod.block(shape[0], shape[1], shape[2], value, map) or mod.block(shape[0], shape[1], shape[2], map)
    else:
        raise ValueError, 'Unsupported shape %s'%shape
    
def subblock(parent, slice):
    """Create a subblock of 'parent'.

    Arguments:

      parent: the parent block to reference.
      slice: the slice of the parent to refer to.
    """

    dtype = parent.dtype
    mod = import_module('vsip.parallel.block', dtype)

    if len(slice) == 1:
        return mod.subblock(parent, slice[0])
    elif len(slice) == 2:
        return mod.subblock(parent, slice[0], slice[1])
    elif len(slice) == 3:
        return mod.subblock(parent, slice[0], slice[1], slice[2])
    else:
        raise ValueError, 'Unsupported slice %s'%slice

