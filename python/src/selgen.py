#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import vector
import numpy

def ramp(start, increment, length, dtype=float):
    """Produce a vector whose values are v[i] = start + increment * i."""

    a = start + increment * numpy.arange(length, dtype=dtype)
    return vector.vector(array=a)

def first(j, f, v, w):
    """Return the first index i >= j where f(v[j], w[j]) returns True."""

    for i in range(j, v.size()):
        if f(v[i], w[i]):
            return i
    return v.size()
