from numpy import array, arange
from vsip import matrix

# Create matrix from scratch
m = matrix(float, 4, 4)
# Access array
a = m.array()
m[0, 0] = 1
# Make sure m and a are referring to the same memory.
assert m[0,0] == a[0,0] == 1

# Create 1D float array
a = arange(16, dtype=float)
# Reshape into 2D array
a.shape = (4,4)
# Wrap it in a matrix
m = matrix(a)
m[0,0] = 3
assert m[0,0] == a[0,0] == 3

# Test rows access
assert (m[1:3] == a[1:3]).all()
assert (m[1:3:2] == a[1:3:2]).all()
assert (m[3:1:-2] == a[3:1:-2]).all()
assert (m[1::2] == a[1::2]).all()
assert (m[:3:2] == a[:3:2]).all()
assert (m[1:-2:2] == a[1:-2:2]).all()
assert (m[-1::-2] == a[-1::-2]).all()

# Test column slice access
assert (m[1:3,-2] == a[1:3,-2]).all()
assert (m[1:3:2,-2] == a[1:3:2,-2]).all()
assert (m[3:1:-2,-2] == a[3:1:-2,-2]).all()
assert (m[1::2,-2] == a[1::2,-2]).all()
assert (m[:3:2,-2] == a[:3:2,-2]).all()
assert (m[1:-2:2,-2] == a[1:-2:2,-2]).all()
assert (m[-1::-2,-2] == a[-1::-2,-2]).all()

# Test row slice access
assert (m[1,1:3] == a[1,1:3]).all()
assert (m[1,1:3:2] == a[1,1:3:2]).all()
assert (m[1,3:1:-2] == a[1,3:1:-2]).all()
assert (m[1,1::2] == a[1,1::2]).all()
assert (m[1,:3:2] == a[1,:3:2]).all()
assert (m[1,1:-2:2] == a[1,1:-2:2]).all()
assert (m[1,-1::-2] == a[1,-1::-2]).all()

# Test submatrix access
assert (m[1:3,2:4] == a[1:3,2:4]).all()
assert (m[1:3:2,2:4:2] == a[1:3:2,2:4:2]).all()
assert (m[3:1:-2,3:1:-2] == a[3:1:-2,3:1:-2]).all()
assert (m[1::2,2::2] == a[1::2,2::2]).all()
assert (m[:3:2,:-1:2] == a[:3:2,:-1:2]).all()
assert (m[1:-2:2,1:-1:2] == a[1:-2:2,1:-1:2]).all()
assert (m[-1::-2,-1::-2] == a[-1::-2,-1::-2]).all()
