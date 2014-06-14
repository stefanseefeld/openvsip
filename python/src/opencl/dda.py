#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy

sync_in = 1
sync_out = 2
sync_inout = 3

def _import_dda_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip.opencl', globals(), locals(), ['fdda'], -1) 
        mod = _temp.fdda
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.opencl', globals(), locals(), ['ddda'], -1) 
        mod = _temp.ddda
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

def dda(block, sync_policy):
    """Create a Direct Data Access object for the given block."""

    mod = _import_dda_module(block.dtype)

    # Create a new block
    if len(block.shape) == 1:
        return mod.dda1(block, sync_policy)
    elif len(block.shape) == 2:
        return mod.dda2(block, sync_policy)
    else:
        raise ValueError, 'Unsupported shape %s'%shape
