#
# Copyright (c) 2015 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy

suffixes = {numpy.float32: '_f',
            numpy.float64: '_d',
            numpy.complex128: '_cd',
            numpy.complex64: '_cf',
            numpy.int32: '_i',
            float: '_d',
            complex: '_cd',
            int: '_i',
            bool: '_b'}

def import_module(name, dtype):
    """Import a module whose name is suffixed with a type encoding"""
    
    package, name = name.rsplit('.', 1)
    mod = None
    if type(dtype) is numpy.dtype:
        dtype = dtype.type
    if dtype in suffixes:
        mod = '_%s%s'%(name, suffixes[dtype])
        _temp = __import__(package, globals(), locals(), [mod], -1) 
        mod = getattr(_temp, mod)
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

