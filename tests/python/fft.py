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
from vsip.signal.fft import *
import numpy as np
#from matplotlib.pyplot import *

v1 = ramp(float, 0, 0.1, 1024)
v1 = elm.sin(v1)
fwd_fft = fft(float, fwd, 1024, 1., 1, alg_hint.time)
inv_fft = fft(float, inv, 1024, 1./1024, 1, alg_hint.time)
v2 = vector(complex, 513)
fwd_fft(v1, v2)
v3 = vector(float, 1024)
inv_fft(v2, v3)

assert np.isclose(v1, v3).all()

#plot(v1)
#plot(v3)
#show()
