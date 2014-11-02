#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip._cast import as_type
from vsip._block import _import_block_module

def cast(dtype, v):

    mod = _import_block_module(dtype)
    return v.__class__(block=as_type(v.block, dtype))
