#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy

def _import_block_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip', globals(), locals(), ['_block_f'], -1) 
        mod = _temp._block_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip', globals(), locals(), ['_block_d'], -1) 
        mod = _temp._block_d
    elif dtype == complex:
        _temp = __import__('vsip', globals(), locals(), ['_block_cd'], -1) 
        mod = _temp._block_cd
    elif dtype == int:
        _temp = __import__('vsip', globals(), locals(), ['_block_i'], -1) 
        mod = _temp._block_i
    elif dtype == bool:
        _temp = __import__('vsip', globals(), locals(), ['_block_b'], -1) 
        mod = _temp._block_b
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

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

    mod = _import_block_module(dtype)

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
    mod = _import_block_module(dtype)

    if len(slice) == 1:
        return mod.subblock(parent, slice[0])
    elif len(slice) == 2:
        return mod.subblock(parent, slice[0], slice[1])
    elif len(slice) == 3:
        return mod.subblock(parent, slice[0], slice[1], slice[2])
    else:
        raise ValueError, 'Unsupported slice %s'%slice

