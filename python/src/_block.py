#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy

def import_block_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip', globals(), locals(), ['fblock'], -1) 
        mod = _temp.fblock
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip', globals(), locals(), ['dblock'], -1) 
        mod = _temp.dblock
    elif dtype == complex:
        _temp = __import__('vsip', globals(), locals(), ['cdblock'], -1) 
        mod = _temp.cdblock
    elif dtype == int:
        _temp = __import__('vsip', globals(), locals(), ['iblock'], -1) 
        mod = _temp.cdblock
    elif dtype == bool:
        _temp = __import__('vsip', globals(), locals(), ['bblock'], -1) 
        mod = _temp.cdblock
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

def block(array=None, dtype=None, shape=None):
    """Create a block.

    Valid argument combinations:

      (array): an existing array-like container to wrap.
      (dtype, shape): element-type and shape for a new block.
    """

    # First figure out the data type so we know which
    # module to import.
    if array is not None:
        # make sure we have all the required metadata
        array = numpy.array(array, copy=False)
        dtype = array.dtype
    else:
        assert dtype and shape

    mod = import_block_module(dtype)

    if array is not None:
        return mod.block(array)
        
    # Create a new block
    if len(shape) == 1:
        return mod.block(shape[0])
    elif len(shape) == 2:
        return mod.block(shape[0], shape[1])
    elif len(shape) == 3:
        return mod.block(shape[0], shape[1], shape[2])
    else:
        raise ValueError, 'Unsupported shape %s'%domain
    
def subblock(parent, slice):
    """Create a subblock of 'parent'.

    Arguments:

      parent: the parent block to reference.
      slice: the slice of the parent to refer to.
    """

    dtype = parent.dtype
    mod = import_block_module(dtype)

    if len(slice) == 1:
        return mod.subblock(parent, slice[0])
    elif len(slice) == 2:
        return mod.subblock(parent, slice[0], slice[1])
    elif len(slice) == 3:
        return mod.subblock(parent, slice[0], slice[1], slice[2])
    else:
        raise ValueError, 'Unsupported slice %s'%slice

