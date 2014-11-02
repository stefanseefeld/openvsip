#
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import vector 
from vsip import matrix
from vsip.selgen.generation import ramp
from vsip.math import elementwise as elm
from vsip.signal import *
from vsip.signal.fftm import *
import numpy as np
#from matplotlib.pyplot import *

v1 = ramp(float, 0, 0.1, 1024)
v1 = elm.sin(v1)
m1 = matrix(float, 16, 1024)
for r in range(16):
    m1[r,:] = ramp(float, r, 0.1, 1024)
fwd_fftm = fftm(float, fwd, 16, 1024, 1., 0, 1, alg_hint.time)
inv_fftm = fftm(float, inv, 16, 1024, 1./1024, 0, 1, alg_hint.time)
m2 = matrix(complex, 16, 513)
fwd_fftm(m1, m2)
m3 = matrix(float, 16, 1024)
inv_fftm(m2, m3)

assert np.isclose(m1, m3).all()

#plot(v1)
#plot(v3)
#show()
