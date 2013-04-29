#! /usr/bin/env python

from numpy import array, arange
from vsip import vector
from vsip.cuda.compiler import SourceModule

# Create 1D float array
a = arange(8, dtype=float)
# Wrap it in a vector
input = vector(a)
print input.array()
output = vector(float, 8)

mod = SourceModule("""
    __global__ void doublify(float *in, float *out)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      out[idx] = 2 * in[idx];
    }
    """)

func = mod.get_function("doublify")
func(input, output, block=(4,4,1))

print "original array:"
print input.array()
print "doubled with kernel:"
print output.array()
