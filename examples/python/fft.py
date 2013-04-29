#!/usr/bin/env python

""" Description
      Define simple step-function and display its fourier transform."""

from vsip import vector
from vsip.signal import *
from vsip.signal.fft import fft
from matplotlib.pyplot import *
import numpy

fft_fwd = fft(numpy.float64, fwd, 1000, 1., 0, alg_hint.time)
input = vector(numpy.float64, 1000)
input[100:] = 0
input[:100] = 1

output = vector(complex, 501)
fft_fwd(input, output)
plot(output)
show()
