#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

from vsip import vector, matrix

def _import_module(dtype):
    import numpy
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip.math', globals(), locals(), ['_matvec_f'], -1) 
        mod = _temp._matvec_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math', globals(), locals(), ['_matvec_d'], -1) 
        mod = _temp._matvec_d
    elif dtype == complex:
        _temp = __import__('vsip.math', globals(), locals(), ['_matvec_cd'], -1) 
        mod = _temp._matvec_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod

def cvjdot(a, b):
    """Compute the conjugate dot product of a and b.
    cvjdot(a, b) = a^T b^*"""

    pass

def dot(a, b):
    """Compute the dot product of a and b.
    dot(a, b) = a * b"""

    b1 = a.block
    b2 = b.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        return mod.dot(b1, b2)
    else:
        import operator
        import numpy
        return reduce(operator.add, map(operator.mul, numpy.array(b1), numpy.array(b2)))

def trans(m):
    """Return the transpose of matrix 'm'"""
    
    mod = _import_module(m.block.dtype)
    return matrix(array=mod.trans(m.block))

def herm(m):
    """Return the Hermitian, i.e. the conjugate transpose of matrix 'm'"""

    mod = _import_module(m.block.dtype)
    return matrix(array=mod.herm(m.block))

def kron(alpha, v, w):
    """Return the Kronecker product of v and w."""

    b1 = v.block
    b2 = w.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        return mod.kron(alpha, b1, b2)
    else:
        import numpy
        return numpy.kron(b1, b2)

def outer(alpha, v, w):
    """Return the outer product of v and w."""

    b1 = v.block
    b2 = w.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        return mod.outer(alpha, b1, b2)
    else:
        pass

def prod(v, w):
    """Return the matrix product of v and w."""

    b1 = v.block
    b2 = w.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        b3 = mod.prod(b1, b2)
        if len(b3.shape) == 2:
            return matrix(block=b3)
        else:
            return vector(block=b3)
    else:
        import numpy
        if len(b1.shape) == len(b2.shape):
            return matrix(array=numpy.tensordot(array(b1), array(b2), (1,0)))
        elif len(b1.shape) == 1:
            return vector(array=numpy.tensordot(array(b1), array(b2), (0,0)))
        else:
            return vector(array=numpy.tensordot(array(b1), array(b2), (1,0)))

def prodh(v, w):
    """Return the matrix product of v and the Hermitian of w."""

    b1 = v.block
    b2 = w.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        return matrix(block=mod.prodh(b1, b2))
    else:
        import numpy
        mod = _import_module(b2.dtype)
        return matrix(array=numpy.tensordot(array(b1), array(mod.herm(b2)), (1,0)))

def prodj(v, w):
    """Return the matrix product of v and the Hermitian of w."""

    b1 = v.block
    b2 = w.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        return matrix(block=mod.prodj(b1, b2))
    else:
        import numpy
        mod = _import_module(b2.dtype)
        return matrix(array=numpy.tensordot(array(b1), array(mod.conj(b2)), (1,0)))

def prodt(v, w):
    """Return the matrix product of v and the transpose of w."""

    b1 = v.block
    b2 = w.block
    if b1.dtype == b2.dtype:
        mod = _import_module(b1.dtype)
        return matrix(block=mod.prodt(b1, b2))
    else:
        import numpy
        mod = _import_module(b2.dtype)
        return matrix(array=numpy.tensordot(array(b1), array(mod.trans(b2)), (1,0)))

def gemp(alpha, a, b, beta, c, opa=None, opb=None):
    """Return the generalized matrix product c = alpha*opa(a)*opb(b) + beta*c."""

    pass

def gems(alpha, a, beta, c, opa=None):
    """Return the generalized matrix product c = alpha*opa(a) + beta*c."""

    pass

def vmmul(axis, v, m):
    """Return the elementwise multiplication of m with the replication of v."""

    vb = v.block
    mb = m.block
    if vb.dtype == mb.dtype:
        mod = _import_module(vb.dtype)
        return matrix(block=mod.vmmul(axis, vb, mb))
    else:
        pass

def cumsum(v):
    """Return the cumulative sum of elements in v."""

    pass


