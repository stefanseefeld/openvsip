/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   Helper functions for device-side complex arithmetic.

#ifndef VSIP_OPT_CUDA_KERNELS_CMPLX_CUH
#define VSIP_OPT_CUDA_KERNELS_CMPLX_CUH 1

#include <cuComplex.h>
#include <math_constants.h>
// overloaded '*' for cuComplex
__device__ inline cuComplex operator*(cuComplex b, cuComplex c)
{
  cuComplex a;

  a.x = b.x * c.x - b.y * c.y;
  a.y = b.y * c.x + b.x * c.y;

  return a;
}

// overloaded '*=' for cuComplex *= float
//  b = b * c
__device__ inline cuComplex& operator*=(cuComplex& b, float c)
{
  b.x *= c;
  b.y *= c;

  return b;
}

// overloaded '+=' for cuComplex
__device__ inline cuComplex& operator+=(cuComplex& b, cuComplex c)
{
  b.x += c.x;
  b.y += c.y;

  return b;
}

// overloaded '/=' for cuComplex /= float
__device__ inline cuComplex& operator/=(cuComplex& b, float c)
{
  b.x /= c;
  b.y /= c;

  return b;
}

namespace dev
{

// magnitude of complex number
//  c = |a|^2
__device__ inline void cmagsq(float& c, cuComplex a)
{
  c = a.x * a.x + a.y * a.y;
}

// complex multiply with scale
//   d = (a * b) * c   where a, b and d are complex and c is real
__device__ inline void cmuls(cuComplex& d, cuComplex a, cuComplex b, float c)
{
  d.x = (a.x * b.x - a.y * b.y) * c;
  d.y = (a.y * b.x + a.x * b.y) * c;
}

// complex multiply
//   c = a * b   where a, b and c are complex
//   No use of intrinsic functions here since they result in increased
//   number of instructions.  
__device__ inline void cmul(cuComplex& c, cuComplex a, cuComplex b)
{
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.y * b.x + a.x * b.y;
}

// complex multiply with complex conjugate of the second argument
//  c = a * conj(b)
__device__ inline void cmulc(cuComplex& c, cuComplex a, cuComplex b)
{
  c.x = __fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
  c.y = __fmul_rn(a.y, b.x) - __fmul_rn(a.x, b.y);
}

// complex multiply with optional complex conjugate of the second argument
// (primarily for use with convolution/correlation functions).
//   c = a * b or c = a * conj(b) if 'factor' is +1 or -1 respectively
__device__ inline void cmulco(cuComplex& c, cuComplex a, cuComplex b, float factor)
{
  c.x = a.x * b.x - factor * a.y * b.y;
  c.y = a.y * b.x + factor * a.x * b.y;
}

// scalar-complex multiply
//   c = a * b   where b and c are complex and a is real
__device__ inline void scmul(cuComplex& c, float a, cuComplex b)
{
  c.x = a * b.x;
  c.y = a * b.y;
}

// scalar-complex multiply-add
//   c += a * b   where b and c are complex and a is real
__device__ inline void scmadd(cuComplex& c, float a, cuComplex b)
{
  c.x += a * b.x;
  c.y += a * b.y;
}

// complex conjugate
//   c = conj(a)    where a and c are complex
__device__ inline void cconj(cuComplex& c, cuComplex a)
{
  c.x =  a.x;
  c.y = -a.y;
}

// complex conjugate (overloaded as a dummy operator for floats)
__device__ inline void cconj(float& c, float a)
{
  c = a;
}

// complex divide by real
//   c = a / b where a, c complex and b real
__device__ inline void cdivr(cuComplex& c, cuComplex a, float b)
{
  c.x = a.x / b;
  c.y = a.y / b;
}

// complex divide
//   c = a / b   where a, b and c are complex
__device__ inline void cdiv(cuComplex& c, cuComplex a, cuComplex b)
{
  float den = b.x*b.x + b.y*b.y;
  cuComplex num;
  cuComplex bcon;
  cconj (bcon, b);
  cmul (num, a, bcon); // num = a * conj(b)
  c.x = num.x / den;
  c.y = num.y / den;
}

// complex log
//   c = log(a)   where a and c are complex
__device__ inline void clog(cuComplex& c, cuComplex a)
{
  float tmp = a.x*a.x + a.y*a.y;
  tmp = logf(tmp);
  c.x = tmp * 0.5f;
  c.y = atan2f(a.y, a.x);
}

// complex log10
//   c = log10(a)   where a and c are complex
__device__ inline void clog10(cuComplex& c, cuComplex a)
{
  float tmp = a.x*a.x + a.y*a.y;
  tmp = log10f(tmp);
  c.x = tmp * 0.5f;
  c.y = atan2f(a.y, a.x) / CUDART_LNT_F;
}

// complex square-root
//   c = sqrt(a) where a and c are complex
__device__ inline void csqrt(cuComplex& c, cuComplex a)
{
  float rpx = a.x + __fsqrt_rn((a.x * a.x) + (a.y * a.y));
  c.x = __fsqrt_rn(0.5 * rpx);
  c.y = a.y / __fsqrt_rn(2.0 * rpx);
}

// complex arccosine
//   c = acos(a) where a and c are complex
__device__ inline void cacos(cuComplex& c, cuComplex a)
{
  cuComplex arg;
  cuComplex temp = a * a;

  temp.x = 1 - temp.x;
  temp.y = -temp.y;
  csqrt(temp, temp);
  arg.x = a.x - temp.y;
  arg.y = a.y + temp.x;
  clog(arg, arg);
  c.x = arg.y;
  c.y = -arg.x;
}

// complex arcsine
//   c = asin(a) where a and c are complex
__device__ inline void casin(cuComplex& c, cuComplex a)
{
  cuComplex arg;
  cuComplex temp = a * a;

  temp.x = 1 - temp.x;
  temp.y = -temp.y;
  csqrt(temp, temp);
  arg.x = -a.y + temp.x;
  arg.y = a.x + temp.y;
  clog(arg, arg);
  c.x = arg.y;
  c.y = -arg.x;
}

// complex hyperbolic cosine
//   c = cosh(a) where a and c are complex
__device__ inline void ccosh(cuComplex& c, cuComplex a)
{
  c.x = coshf(a.x) * cosf(a.y);
  c.y = sinhf(a.x) * sinf(a.y);
}

// complex hyperbolic sine
//   c = sinh(a) where a and c are complex
__device__ inline void csinh(cuComplex& c, cuComplex a)
{
  c.x = sinhf(a.x) * cosf(a.y);
  c.y = coshf(a.x) * sinf(a.y);
}

// complex ceiling
//   c = ceil(a) where a and c are complex
__device__ inline void cceil(cuComplex &c, cuComplex a)
{
  c.x = ceilf(a.x);
  c.y = ceilf(a.y);
}

// Euler function
//   c = euler(a) where a is real and c is complex
__device__ inline void ceuler(cuComplex& c, float a)
{
  c.x = cosf(a);
  c.y = sinf(a);
}

// complex exponential
//   c = exp(a) where a and c are complex
__device__ inline void cexp(cuComplex& c, cuComplex a)
{
  c.x = expf(a.x) * cosf(a.y);
  c.y = expf(a.x) * sinf(a.y);
}

// complex base 10 exponential
//   c = exp10(a) where a and c are complex
__device__ inline void cexp10(cuComplex& c, cuComplex a)
{
  // log(10)
  float const lnten = 2.302585092994046;
  c.x = exp10f(a.x) * cosf(a.y * lnten);
  c.y = exp10f(a.x) * sinf(a.y * lnten);
}

// complex floor function
//   c = floor(a) where a and c are complex
__device__ inline void cfloor(cuComplex& c, cuComplex a)
{
  c.x = floorf(a.x);
  c.y = floorf(a.y);
}

// complex reciprocal
//   c = recip(a) where a and c are complex
__device__ inline void crecip(cuComplex& c, cuComplex a)
{
  float mag, arg;
  mag = sqrtf(a.x * a.x + a.y * a.y);
  arg = atan2f(a.y, a.x);

  c.x = (1 / mag) * cosf(arg);
  c.y = -(1 / mag) * sinf(arg);
}

// complex reciprocal square root
//   c = rsqrt(a) where a and c are complex
__device__ inline void crsqrt(cuComplex& c, cuComplex a)
{
  float mag, arg;
  mag = sqrtf(a.x * a.x + a.y * a.y);
  arg = atan2f(a.y, a.x);

  c.x = rsqrtf(mag) * cosf(arg / 2);
  c.y = -rsqrtf(mag) * sinf(arg / 2);
}

// complex square
//   c = square(a) where a and c are complex
__device__ inline void csq(cuComplex& c, cuComplex a)
{
  c.x = a.x * a.x - a.y * a.y;
  c.y = 2 * a.x * a.y;
}

// complex tangent
//   c = tan(a) where a and c are complex
__device__ inline void ctan(cuComplex& c, cuComplex a)
{
  float denominator = cosf(a.x) * cosf(a.x) + sinhf(a.y) * sinhf(a.y);

  c.x = sinf(a.x) * cosf(a.x) / denominator;
  c.y = sinhf(a.y) * coshf(a.y) / denominator;
}

// complex hyperbolic tangent
//   c = tanh(a) where a and c are complex
__device__ inline void ctanh(cuComplex& c, cuComplex a)
{
  float denominator = coshf(2 * a.x) + cosf(2 * a.y);

  c.x = sinhf(2 * a.x) / denominator;
  c.y = sinf(2 * a.y) / denominator;
}

/// complex cosine
///   c = cos(a) where a and c are complex
__device__ inline void ccos(cuComplex& c, cuComplex a)
{
  float sinr, cosr, sinhi, coshi;
  __sincosf(a.x, &sinr, &cosr);
  sinhi = sinhf(a.y);
  coshi = coshf(a.y);
  c.x =   cosr * coshi;
  c.y = -(sinr * sinhi);
}

/// complex sine
///   c = sin(a) where a and c are complex
__device__ inline void csin(cuComplex& c, cuComplex a)
{
  float sinr, cosr, sinhi, coshi;
  __sincosf(a.x, &sinr, &cosr);
  sinhi = sinhf(a.y);
  coshi = coshf(a.y);
  c.x = sinr * coshi;
  c.y = cosr * sinhi;
}
} // namespace dev

#endif //  VSIP_OPT_CUDA_KERNELS_CMPLX_CUH
