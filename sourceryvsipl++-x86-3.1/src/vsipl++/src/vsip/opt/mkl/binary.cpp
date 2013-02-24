/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   MKL bindings for binary expressions.

#include <vsip/opt/mkl/binary.hpp>
#include <mkl_vml.h>

namespace vsip
{
namespace impl
{
namespace mkl
{

#define BINARY(FUN, IN1, IN2, OUT, MKLF, MKLIN1, MKLIN2, MKLOUT)\
void FUN(IN1 in1, IN2 in2, OUT out, length_type len)		\
{								\
  MKLF(static_cast<int>(len),					\
       reinterpret_cast<MKLIN1>(in1),				\
       reinterpret_cast<MKLIN2>(in2),				\
       reinterpret_cast<MKLOUT>(out));				\
}

BINARY(add, float const *, float const *, float *,
       vsAdd, float const *, float const *, float *);
BINARY(add, double const *, double const *, double *,
       vdAdd, double const *, double const *, double *);
BINARY(add, complex<float> const *, complex<float> const *, complex<float> *,
       vcAdd, MKL_Complex8 const *, MKL_Complex8 const *, MKL_Complex8 *);
BINARY(add, complex<double> const *, complex<double> const *, complex<double> *,
       vzAdd, MKL_Complex16 const *, MKL_Complex16 const *, MKL_Complex16 *);

BINARY(sub, float const *, float const *, float *,
       vsSub, float const *, float const *, float *);
BINARY(sub, double const *, double const *, double *,
       vdSub, double const *, double const *, double *);
BINARY(sub, complex<float> const *, complex<float> const *, complex<float> *,
       vcSub, MKL_Complex8 const *, MKL_Complex8 const *, MKL_Complex8 *);
BINARY(sub, complex<double> const *, complex<double> const *, complex<double> *,
       vzSub, MKL_Complex16 const *, MKL_Complex16 const *, MKL_Complex16 *);

BINARY(mul, float const *, float const *, float *,
       vsMul, float const *, float const *, float *);
BINARY(mul, double const *, double const *, double *,
       vdMul, double const *, double const *, double *);
BINARY(mul, complex<float> const *, complex<float> const *, complex<float> *,
       vcMul, MKL_Complex8 const *, MKL_Complex8 const *, MKL_Complex8 *);
BINARY(mul, complex<double> const *, complex<double> const *, complex<double> *,
       vzMul, MKL_Complex16 const *, MKL_Complex16 const *, MKL_Complex16 *);

BINARY(div, float const *, float const *, float *,
       vsDiv, float const *, float const *, float *);
BINARY(div, double const *, double const *, double *,
       vdDiv, double const *, double const *, double *);
BINARY(div, complex<float> const *, complex<float> const *, complex<float> *,
       vcDiv, MKL_Complex8 const *, MKL_Complex8 const *, MKL_Complex8 *);
BINARY(div, complex<double> const *, complex<double> const *, complex<double> *,
       vzDiv, MKL_Complex16 const *, MKL_Complex16 const *, MKL_Complex16 *);

BINARY(pow, float const *, float const *, float *,
       vsPow, float const *, float const *, float *);
BINARY(pow, double const *, double const *, double *,
       vdPow, double const *, double const *, double *);
BINARY(pow, complex<float> const *, complex<float> const *, complex<float> *,
       vcPow, MKL_Complex8 const *, MKL_Complex8 const *, MKL_Complex8 *);
BINARY(pow, complex<double> const *, complex<double> const *, complex<double> *,
       vzPow, MKL_Complex16 const *, MKL_Complex16 const *, MKL_Complex16 *);

BINARY(hypot, float const *, float const *, float *,
       vsHypot, float const *, float const *, float *);
BINARY(hypot, double const *, double const *, double *,
       vdHypot, double const *, double const *, double *);

BINARY(atan2, float const *, float const *, float *,
       vsAtan2, float const *, float const *, float *);
BINARY(atan2, double const *, double const *, double *,
       vdAtan2, double const *, double const *, double *);


#undef BINARY

} // namespace vsip::impl::mkl
} // namespace vsip::impl
} // namespace vsip

 
