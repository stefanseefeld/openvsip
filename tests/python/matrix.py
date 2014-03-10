from numpy import array, arange
from vsip import matrix

# Create matrix from scratch
m = matrix(dtype=float, rows=4, cols=4)
# Access array
a = array(m, copy=False)
m[0, 0] = 1
# Make sure m and a are referring to the same memory.
assert m[0,0] == a[0,0] == 1

# Create 1D float array
a = arange(16, dtype=float)
# Reshape into 2D array
a.shape = (4,4)
# Wrap it in a matrix
m = matrix(array=a)
m[0,0] = 3
assert m[0,0] == a[0,0] == 3

# Test rows access
assert array(m[1:3] == a[1:3]).all()
assert array(m[1:3:2] == a[1:3:2]).all()
assert array(m[3:1:-2] == a[3:1:-2]).all()
assert array(m[1::2] == a[1::2]).all()
assert array(m[:3:2] == a[:3:2]).all()
assert array(m[1:-2:2] == a[1:-2:2]).all()
assert array(m[-1::-2] == a[-1::-2]).all()

# Test column slice access
assert array(m[1:3,-2] == a[1:3,-2]).all()
assert array(m[1:3:2,-2] == a[1:3:2,-2]).all()
assert array(m[3:1:-2,-2] == a[3:1:-2,-2]).all()
assert array(m[1::2,-2] == a[1::2,-2]).all()
assert array(m[:3:2,-2] == a[:3:2,-2]).all()
assert array(m[1:-2:2,-2] == a[1:-2:2,-2]).all()
assert array(m[-1::-2,-2] == a[-1::-2,-2]).all()

# Test row slice access
assert array(m[1,1:3] == a[1,1:3]).all()
assert array(m[1,1:3:2] == a[1,1:3:2]).all()
assert array(m[1,3:1:-2] == a[1,3:1:-2]).all()
assert array(m[1,1::2] == a[1,1::2]).all()
assert array(m[1,:3:2] == a[1,:3:2]).all()
assert array(m[1,1:-2:2] == a[1,1:-2:2]).all()
assert array(m[1,-1::-2] == a[1,-1::-2]).all()

# Test submatrix access
assert array(m[1:3,2:4] == a[1:3,2:4]).all()
assert array(m[1:3:2,2:4:2] == a[1:3:2,2:4:2]).all()
assert array(m[3:1:-2,3:1:-2] == a[3:1:-2,3:1:-2]).all()
assert array(m[1::2,2::2] == a[1::2,2::2]).all()
assert array(m[:3:2,:-1:2] == a[:3:2,:-1:2]).all()
assert array(m[1:-2:2,1:-1:2] == a[1:-2:2,1:-1:2]).all()
assert array(m[-1::-2,-1::-2] == a[-1::-2,-1::-2]).all()

# Test slice assignment
# column subvectors
a = array(m, copy=True)
m[1:3] = 2.
a[1:3] = 2.
assert array(m == a).all()
m[1:5:2] = 3.
a[1:5:2] = 3.
assert array(m == a).all()
m[5:1:-2] = 1.
a[5:1:-2] = 1.
assert array(m == a).all()
m[1::2] = 42.
a[1::2] = 42.
assert array(m == a).all()
m[:5:2] = 13.
a[:5:2] = 13.
assert array(m == a).all()
m[1:-2:2] = 12.
a[1:-2:2] = 12.
assert array(m == a).all()
m[-1::-2] = 11.
a[-1::-2] = 11.
assert array(m == a).all()
# row subvectors
m[:,1:3] = 2.
a[:,1:3] = 2.
assert array(m == a).all()
m[:,1:5:2] = 3.
a[:,1:5:2] = 3.
assert array(m == a).all()
m[:,5:1:-2] = 1.
a[:,5:1:-2] = 1.
assert array(m == a).all()
m[:,1::2] = 42.
a[:,1::2] = 42.
assert array(m == a).all()
m[:,:5:2] = 13.
a[:,:5:2] = 13.
assert array(m == a).all()
m[:,1:-2:2] = 12.
a[:,1:-2:2] = 12.
assert array(m == a).all()
m[:,-1::-2] = 11.
a[:,-1::-2] = 11.
assert array(m == a).all()

# TBD: Test compound assignment
