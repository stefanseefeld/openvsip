from vsip import matrix
from vsip.math.solvers import svd
from vsip.math.matvec import prod, trans

import numpy as np

A = matrix(array=[[1.,2.,3.],[4.,5.,6.]])
M,N = A.rows(), A.cols()
print A

svd = svd(float, M, N, svd.uvfull, svd.uvfull)

s = svd.decompose(A)
S = matrix(float, M, N)

S.diag()[:] = s
U = svd.u(0, M-1)
V = svd.v(0, N-1)
C = prod(prod(U, S), trans(V))
print C
assert np.isclose(C, A).all()
