#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy as np
from scipy import linalg
from vsip import vector, matrix
from vsip import random
from vsip.selgen import generation as gen
from vsip.math import elementwise as elm
from vsip.math import reductions as red
from vsip.math.solvers import llsqsol
#import matplotlib.pyplot as plt


# Define 'A' to be a two-column matrix containing Y[X] and X
A = matrix(float, 10, 2)
X = gen.ramp(float, 0.1, 0.1, 10)
A[:,0] = elm.exp(-X)
A[:,1] = X

c1,c2= 5.0,2.0
Y = c1*elm.exp(-X) + c2*X

Z = matrix(float, 10, 1)
Z[:,0] = Y + 0.05*red.maxval(Y)[0]*random.rand(float).randn(Y.length())

R = llsqsol(A, Z)
c,resid,rank,sigma = linalg.lstsq(A, Z)

# Compare results of llsqsol with results from linalg.lstsq
assert np.isclose(R, c).all()


#X2 = gen.ramp(float, 0.1, 0.9/100, 100) 
#Y2 = c[0]*elm.exp(-X2) + c[1]*X2
#R2 = elm.exp(-X2)*R[0,0] + X2*R[1,0]

#plt.plot(X, Z, 'x', X2, R2)
#plt.plot(X, Z,'x', X2, Y2)
#plt.axis([0,1.1,3.0,5.5])
#plt.xlabel('$x_i$')
#plt.title('Data fitting with llsqsol')
#plt.show()
