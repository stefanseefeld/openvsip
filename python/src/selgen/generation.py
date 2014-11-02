#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import vector

def ramp(dtype, start, increment, length):
    """Produce a vector whose values are :math:`v_i = start + increment * i`."""

    import numpy
    a = start + increment * numpy.arange(length, dtype=dtype)
    return vector(array=a)

