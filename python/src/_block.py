#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import import_module
import numpy

def block(*args, **kwargs):
    """Create a block.

    Valid argument combinations:

      block(dtype, shape): create a block of the given type and shape
      block(dtype, shape, value): create a block of the given type and shape and initialize it to value
      block(array=A): create a block from an existing array-like container
    """

    # First figure out the data type so we know which
    # module to import.
    if args:
        dtype = args[0]
        shape = args[1]
        value = len(args) == 3 and args[2] or None
            
    elif 'array' in kwargs:
        # make sure we have all the required metadata
        array = numpy.array(kwargs['array'], copy=False)
        dtype = array.dtype

    mod = import_module('vsip.block', dtype)

    if 'array' in kwargs:
        return mod.block(array)
        
    # Create a new block
    if len(shape) == 1:
        return value and mod.block(shape[0], value) or mod.block(shape[0])
    elif len(shape) == 2:
        return value and mod.block(shape[0], shape[1], value) or mod.block(shape[0], shape[1])
    elif len(shape) == 3:
        return value and mod.block(shape[0], shape[1], shape[2], value) or mod.block(shape[0], shape[1], shape[2])
    else:
        raise ValueError, 'Unsupported shape %s'%shape
    
def subblock(parent, slice):
    """Create a subblock of 'parent'.

    Arguments:

      parent: the parent block to reference.
      slice: the slice of the parent to refer to.
    """

    dtype = parent.dtype
    mod = import_module('vsip.block', dtype)

    if len(slice) == 1:
        return mod.subblock(parent, slice[0])
    elif len(slice) == 2:
        return mod.subblock(parent, slice[0], slice[1])
    elif len(slice) == 3:
        return mod.subblock(parent, slice[0], slice[1], slice[2])
    else:
        raise ValueError, 'Unsupported slice %s'%slice

