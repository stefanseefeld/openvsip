#
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import vector 
from vsip.signal.window import *
from scipy.signal import get_window as ref_window
import numpy as np
from matplotlib.pyplot import *

# Attention: For now these "tests" are only used in a qualitative way.
#
# TODO: The SciPy reference implementations of these window functions
#       appear to be using slightly different formulae, causing significant
#       numeric deviation. Investigate the cause of this, and adjust parameters
#       to get predictable results so we can refine the comparison below.
#       To see the difference, call:
#       plot(v - vector(array=ref))


# blackman
v = blackman(float, 256)
ref = ref_window('blackman', 256)
assert np.isclose(v, ref, 1e-01, 1e-03).all()

# cheby
v = cheby(float, 256, 100)
ref = ref_window(('chebwin', 100), 256)
assert np.isclose(v, ref, 1e-01, 1e-03).all()

# hanning
v = hanning(float, 256)
ref = ref_window('hann', 256)
assert np.isclose(v, ref, 1, 1e-03).all()

# kaiser
v = kaiser(float, 256, 4.)
ref = ref_window(('kaiser', 4.), 256)
assert np.isclose(v, ref, 1e-01, 1e-03).all()


#plot(v)
#show()
#plot(ref)
#show()
