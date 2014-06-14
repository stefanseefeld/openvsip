#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

def cvjdot(a, b):
    """Compute the conjugate dot product of a and b.
    cvjdot(a, b) = a^T b^*"""

    pass

def dot(a, b):
    """Compute the dot product of a and b.
    dot(a, b) = a * b"""

    pass

def trans(m):
    """Return the transpose of matrix 'm'"""

    pass

def herm(m):
    """Return the Hermitian, i.e. the conjugate transpose of matrix 'm'"""

    pass

def kron(alpha, v, w):
    """Return the Kronecker product of v and w."""

    pass

def outer(alpha, v, w):
    """Return the outer product of v and w."""

    pass

def prod(v, w):
    """Return the matrix product of v and w."""

    pass

def prodh(v, w):
    """Return the matrix product of v and the Hermitian of w."""

    pass

def prodj(v, w):
    """Return the matrix product of v and the conjugate of w."""

    pass

def prodt(v, w):
    """Return the matrix product of v and the transpose of w."""

    pass

def gemp(alpha, a, b, beta, c, opa=None, opb=None):
    """Return the generalized matrix product c = alpha*opa(a)*opb(b) + beta*c."""

    pass

def gems(alpha, a, beta, c, opa=None):
    """Return the generalized matrix product c = alpha*opa(a) + beta*c."""

    pass

def vmmul(v, m, axis=0):
    """Return the elementwise multiplication of m with the replication of v."""

    pass

def cumsum(v):
    """Return the cumulative sum of elements in v."""

    pass


