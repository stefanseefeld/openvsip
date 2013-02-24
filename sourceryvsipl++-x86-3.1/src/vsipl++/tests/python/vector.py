from numpy import array, arange
from vsip import vector

# Create vector from scratch
v = vector(float, 8)
# Access array
a = v.array()
v[0] = 3
# Make sure v and a are referring to the same memory.
assert v[0] == a[0] == 3

# Create 1D float array
a = arange(8, dtype=float)
# Wrap it in a vector
v = vector(a)
v[0] = 3
assert v[0] == a[0] == 3

# Test slicing
assert (v[1:3] == a[1:3]).all()
assert (v[1:5:2] == a[1:5:2]).all()
assert (v[5:1:-2] == a[5:1:-2]).all()
assert (v[1::2] == a[1::2]).all()
assert (v[:5:2] == a[:5:2]).all()
assert (v[1:-2:2] == a[1:-2:2]).all()
assert (v[-1::-2] == a[-1::-2]).all()
