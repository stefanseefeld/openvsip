#
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import vector 
from vsip import matrix
from vsip.selgen.generation import ramp
from vsip.math.matvec import dot, prod
import numpy as np
from numpy import array as A


v1 = ramp(float, 0, 1, 8)
v2 = ramp(float, 1, 2, 8)
d = dot(v1, v2)
assert d == np.dot(A(v1), A(v2))

v1 = vector(array=np.arange(4, dtype=float))
m1 = matrix(array=np.arange(16, dtype=float).reshape(4,4))
m2 = matrix(array=np.arange(16, dtype=float).reshape(4,4))
# vector * matrix
v3 = prod(v1, m1)
assert (A(v3) == np.tensordot(A(v1), A(m1), (0, 0))).all()
# matrix * vector
v3 = prod(m1, v1)
assert (A(v3) == np.tensordot(A(m1), A(v1), (1, 0))).all()
# matrix * matrix
m3 = prod(m1, m2)
assert (A(m3) == np.tensordot(A(m1), A(m2), (1, 0))).all()

