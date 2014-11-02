#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.

import numpy
from vsip import vector, matrix

def _import_module(dtype):
    mod = None
    if dtype == numpy.float32:
        _temp = __import__('vsip.math', globals(), locals(), ['_elementwise_f'], -1) 
        mod = _temp._elementwise_f
    elif dtype in (float, numpy.float64):
        _temp = __import__('vsip.math', globals(), locals(), ['_elementwise_d'], -1) 
        mod = _temp._elementwise_d
    elif dtype == complex:
        _temp = __import__('vsip.math', globals(), locals(), ['_elementwise_cd'], -1) 
        mod = _temp._elementwise_cd
    if not mod:
        raise ValueError, 'Unsupported dtype %s'%(dtype)
    return mod


def unary(func, a):

    b = a.block
    m = _import_module(b.dtype)
    f = getattr(m, func)
    v = a.__class__
    return v(block=f(b))

def binary(func, a1, a2):

    b1 = a1.block
    b2 = a2.block
    if b1.dtype == b2.dtype:
        m = _import_module(b1.dtype)
    else:
        raise ValueError, 'Unsupported combination of dtypes (%s, %s)'%(b1.dtype, b2.dtype)
    f = getattr(m, func)
    v = a1.__class__
    return v(block=f(b1, b2))

def ternary(func, a1, a2, a3):

    b1 = a1.block
    b2 = a2.block
    b3 = a3.block
    if b1.dtype == b2.dtype == b3.dtype:
        m = _import_module(b1.dtype)
    else:
        raise ValueError, 'Unsupported combination of dtypes (%s, %s, %s)'%(b1.dtype, b2.dtype, b3.dtype)
    f = getattr(m, func)
    v = a1.__class__
    return v(block=f(b1, b2, b3))


def acos(a): return unary('acos', a)
def am(a1, a2, a3): return ternary('am', a1, a2, a3)
def arg(a): return unary('arg', a)
def asin(a): return unary('asin', a)
def atan(a): return unary('atan', a)
def atan2(a1, a2): return binary('atan2', a1, a2)
def ceil(a): return unary('ceil', a)
def cos(a): return unary('cos', a)
def cosh(a): return unary('cosh', a)
def euler(a): return unary('euler', a)
def eq(a1, a2): return binary('eq', a1, a2)
def exp(a): return unary('exp', a)
def exp10(a): return unary('exp10', a)
def floor(a): return unary('floor', a)
def ite(a1, a2, a3): return ternary('ite', a1, a2, a3)
def log(a): return unary('log', a)
def log10(a): return unary('log10', a)
def magsq(a): return unary('magsq', a)
def mul(a1, a2): return binary('mul', a1, a2)
def neg(a): return unary('neg', a)
def sin(a): return unary('sin', a)
