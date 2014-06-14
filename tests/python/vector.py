from numpy import array, arange
from vsip import vector

# Create vector from scratch
v = vector(dtype=float, length=8)
# Access as array (by-reference)
a = array(v, copy=False)
v[0] = 3
# Make sure v and a are referring to the same memory.
assert v[0] == a[0] == 3

# Create 1D float array
a = arange(8, dtype=float)
# Wrap it in a vector
v = vector(array=a)
v[0] = 3
a[0] = 3
assert v[0] == 3

# Test slicing access
assert array(v[1:3] == a[1:3]).all()
assert array(v[1:5:2] == a[1:5:2]).all()
assert array(v[5:1:-2] == a[5:1:-2]).all()
assert array(v[1::2] == a[1::2]).all()
assert array(v[:5:2] == a[:5:2]).all()
assert array(v[1:-2:2] == a[1:-2:2]).all()
assert array(v[-1::-2] == a[-1::-2]).all()

# Test slice assignment
a = array(v, copy=True)
v[1:3] = 2.
a[1:3] = 2.
assert array(v == a).all()
v[1:5:2] = 3.
a[1:5:2] = 3.
assert array(v == a).all()
v[5:1:-2] = 1.
a[5:1:-2] = 1.
assert array(v == a).all()
v[1::2] = 42.
a[1::2] = 42.
assert array(v == a).all()
v[:5:2] = 13.
a[:5:2] = 13.
assert array(v == a).all()
v[1:-2:2] = 12.
a[1:-2:2] = 12.
assert array(v == a).all()
v[-1::-2] = 11.
a[-1::-2] = 11.
assert array(v == a).all()

# Test complex manipulation
a = arange(8, dtype=complex)
v = vector(array=a)
a = array(v, copy=True)
v.imag()[:] = 1.
assert array(v.real() == a.real).all()
a.imag = 1.
assert array(v == a).all()
v.real()[:] *= 2.
assert array(v.imag() == a.imag).all()
a.real *= 2.
assert array(v == a).all()

# TBD: Test compound assignment
