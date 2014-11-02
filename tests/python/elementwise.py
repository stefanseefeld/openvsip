#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy as np
from numpy import array
from vsip import vector, matrix
from vsip.math import elementwise as elm

a1 = np.arange(16, dtype=float)
v1 = vector(array=a1)


# Unary functions

assert array(elm.cos(v1) == np.cos(a1)).all()
assert array(elm.sin(v1) == np.sin(a1)).all()

a2 = np.arange(16, dtype=float)
v2 = vector(array=a2)

# Binary functions

assert array(elm.mul(v1, v2) == a1*a2).all()

a3 = np.arange(16, dtype=float)
v3 = vector(array=a3)

# Ternary functions

assert array(elm.am(v1, v2, v3) == (a1+a2)*a3).all()

