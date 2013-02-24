#!/usr/bin/env python

""" Description
      Define simple difference operator and convolve simple input with it."""

from vsip import vector
from vsip.signal import *
from vsip.signal.conv import convolution
from matplotlib.pyplot import *
import numpy

# define differentiation operator
K = vector(numpy.array([-1., 0., 1.]))
# define tics for the X axis
X = vector(numpy.arange(1024, dtype=numpy.float64))
# set up input array
input = vector(numpy.sin(X/numpy.float64(100.)))
# set up output array
output = vector(numpy.float64, 1022)
# create convolution object
conv = convolution(K, symmetry.none, 1024, 1, support_region.min, 0, alg_hint.time)
# run convolution
conv(input, output)
# scale
output *= numpy.float64(50.)
# plot input and output
plot(X, input)
plot(X[:1022], output)
show()
