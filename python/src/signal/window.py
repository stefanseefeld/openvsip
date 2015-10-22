#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import vector
from vsip import import_module

def blackman(dtype, N):
    """Create a vector with blackman window weights.

    arguments:

      :dtype: data type of the vector (e.g. float)
      :N: length of the vector

    .. math::

       y_i = 0.42 * 0.5 * cos(\\frac{2*\pi*i}{N-1}) + 0.08 * cos(\\frac{4*\pi*i}{N-1})"""

    import_module('vsip.block', dtype)
    m = import_module('vsip.signal.window', dtype)
    return vector(block=m.blackman(N))

def cheby(dtype, N, ripple):
    """Create a vector with a Dolph-Chebyshev window of length N.

    arguments:

      :dtype: data type of the vector (e.g. float)
      :N: length of the vector
      :ripple: 

"""

    import_module('vsip.block', dtype)
    m = import_module('vsip.signal.window', dtype)
    return vector(block=m.cheby(N, ripple))

def hanning(dtype, N):
    """Create a vector with Hanning window weights.

    arguments:

      :dtype: data type of the vector (e.g. float)
      :N: length of the vector

    .. math::

       y_i = \\frac{1}{2}(1 - cos(\\frac{2\pi(i+1)}{N+1}))"""


    import_module('vsip.block', dtype)
    m = import_module('vsip.signal.window', dtype)
    return vector(block=m.hanning(N))

def kaiser(dtype, N, beta):
    """Create a vector with Kaiser window weights.

    arguments:

      :dtype: data type of the vector (e.g. float)
      :N: length of the vector

    """

    import_module('vsip.block', dtype)
    m = import_module('vsip.signal.window', dtype)
    return vector(block=m.kaiser(N, beta))

