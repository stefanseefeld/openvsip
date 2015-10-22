#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import import_module
from vsip import vector, matrix

class conv:

    def __init__(self, dtype, symmetry, support, i, decimation, n, hint):

        self.dtype = dtype
        self.symmetry = symmetry
        self.support = support

        m = import_module('vsip.signal.conv', dtype)
        self._impl = m.conv(symmetry, i, decimation, support, n, hint)

    def __call__(self, input, output):

        self._impl(input.block, output.block)

