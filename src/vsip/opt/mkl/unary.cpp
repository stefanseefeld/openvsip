/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   MKL bindings for unary expressions.

#include <vsip/opt/mkl/unary.hpp>
#include <mkl_vml.h>

namespace vsip
{
namespace impl
{
namespace mkl
{

#define UNARY(FUN, IN, OUT, MKLF, MKLIN, MKLOUT)	\
void FUN(IN in, OUT out, length_type len)		\
{							\
  MKLF(static_cast<int>(len),				\
       reinterpret_cast<MKLIN>(in),			\
       reinterpret_cast<MKLOUT>(out));			\
}

UNARY(sq, float const *, float *, vsSqr, float const *, float *);
UNARY(sq, double const *, double *, vdSqr, double const *, double *);

UNARY(conj, complex<float> const *, complex<float> *,
      vcConj, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(conj, complex<double> const *, complex<double> *,
      vzConj, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(mag, float const *, float *, vsAbs, float const *, float *);
UNARY(mag, double const *, double *, vdAbs, double const *, double *);
UNARY(mag, complex<float> const *, float *,
      vcAbs, MKL_Complex8 const *, float *);
UNARY(mag, complex<double> const *, double *,
      vzAbs, MKL_Complex16 const *, double *);

UNARY(arg, complex<float> const *, float *,
      vcArg, MKL_Complex8 const *, float *);
UNARY(arg, complex<double> const *, double *,
      vzArg, MKL_Complex16 const *, double *);

UNARY(sqrt, float const *, float *, vsSqrt, float const *, float *);
UNARY(sqrt, double const *, double *, vdSqrt, double const *, double *);
UNARY(sqrt, complex<float> const *, complex<float> *,
      vcSqrt, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(sqrt, complex<double> const *, complex<double> *,
      vzSqrt, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(exp, float const *, float *, vsExp, float const *, float *);
UNARY(exp, double const *, double *, vdExp, double const *, double *);
UNARY(exp, complex<float> const *, complex<float> *,
      vcExp, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(exp, complex<double> const *, complex<double> *,
      vzExp, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(log, float const *, float *, vsLn, float const *, float *);
UNARY(log, double const *, double *, vdLn, double const *, double *);
UNARY(log, complex<float> const *, complex<float> *,
      vcLn, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(log, complex<double> const *, complex<double> *,
      vzLn, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(log10, float const *, float *, vsLog10, float const *, float *);
UNARY(log10, double const *, double *, vdLog10, double const *, double *);
UNARY(log10, complex<float> const *, complex<float> *, vcLog10,
      MKL_Complex8 const *, MKL_Complex8 *);
UNARY(log10, complex<double> const *, complex<double> *,
      vzLog10, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(sin, float const *, float *, vsSin, float const *, float *);
UNARY(sin, double const *, double *, vdSin, double const *, double *);
UNARY(sin, complex<float> const *, complex<float> *,
      vcSin, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(sin, complex<double> const *, complex<double> *,
      vzSin, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(cos, float const *, float *, vsCos, float const *, float *);
UNARY(cos, double const *, double *, vdCos, double const *, double *);
UNARY(cos, complex<float> const *, complex<float> *,
      vcCos, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(cos, complex<double> const *, complex<double> *,
      vzCos, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(tan, float const *, float *, vsTan, float const *, float *);
UNARY(tan, double const *, double *, vdTan, double const *, double *);
UNARY(tan, complex<float> const *, complex<float> *,
      vcTan, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(tan, complex<double> const *, complex<double> *,
      vzTan, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(acos, float const *, float *, vsAcos, float const *, float *);
UNARY(acos, double const *, double *, vdAcos, double const *, double *);
UNARY(acos, complex<float> const *, complex<float> *,
      vcAcos, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(acos, complex<double> const *, complex<double> *,
      vzAcos, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(asin, float const *, float *, vsAsin, float const *, float *);
UNARY(asin, double const *, double *, vdAsin, double const *, double *);
UNARY(asin, complex<float> const *, complex<float> *,
      vcAsin, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(asin, complex<double> const *, complex<double> *,
      vzAsin, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(atan, float const *, float *, vsAtan, float const *, float *);
UNARY(atan, double const *, double *, vdAtan, double const *, double *);
UNARY(atan, complex<float> const *, complex<float> *,
      vcAtan, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(atan, complex<double> const *, complex<double> *,
      vzAtan, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(sinh, float const *, float *, vsSinh, float const *, float *);
UNARY(sinh, double const *, double *, vdSinh, double const *, double *);
UNARY(sinh, complex<float> const *, complex<float> *,
      vcSinh, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(sinh, complex<double> const *, complex<double> *,
      vzSinh, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(cosh, float const *, float *, vsCosh, float const *, float *);
UNARY(cosh, double const *, double *, vdCosh, double const *, double *);
UNARY(cosh, complex<float> const *, complex<float> *,
      vcCosh, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(cosh, complex<double> const *, complex<double> *,
      vzCosh, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(tanh, float const *, float *, vsTanh, float const *, float *);
UNARY(tanh, double const *, double *, vdTanh, double const *, double *);
UNARY(tanh, complex<float> const *, complex<float> *,
      vcTanh, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(tanh, complex<double> const *, complex<double> *,
      vzTanh, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(asinh, float const *, float *, vsAsinh, float const *, float *);
UNARY(asinh, double const *, double *, vdAsinh, double const *, double *);
UNARY(asinh, complex<float> const *, complex<float> *,
      vcAsinh, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(asinh, complex<double> const *, complex<double> *,
      vzAsinh, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(acosh, float const *, float *, vsAcosh, float const *, float *);
UNARY(acosh, double const *, double *, vdAcosh, double const *, double *);
UNARY(acosh, complex<float> const *, complex<float> *,
      vcAcosh, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(acosh, complex<double> const *, complex<double> *,
      vzAcosh, MKL_Complex16 const *, MKL_Complex16 *);

UNARY(atanh, float const *, float *, vsAtanh, float const *, float *);
UNARY(atanh, double const *, double *, vdAtanh, double const *, double *);
UNARY(atanh, complex<float> const *, complex<float> *,
      vcAtanh, MKL_Complex8 const *, MKL_Complex8 *);
UNARY(atanh, complex<double> const *, complex<double> *,
      vzAtanh, MKL_Complex16 const *, MKL_Complex16 *);

#undef UNARY

} // namespace vsip::impl::mkl
} // namespace vsip::impl
} // namespace vsip

 
