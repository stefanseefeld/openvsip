#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy as np
from numpy import array
from vsip import vector, matrix
from vsip.math import reductions as red

a = np.arange(16, dtype=float)
v = vector(array=a)


assert array(red.meanval(v) == np.mean(a)).all()
assert array(red.maxval(v)[0] == np.max(a)).all()
assert array(red.minval(v)[0] == np.min(a)).all()
assert array(red.sumval(v) == np.sum(a)).all()

