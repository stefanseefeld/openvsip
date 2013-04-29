/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   MKL bindings for unary expressions.

#ifndef vsip_opt_mkl_unary_hpp_
#define vsip_opt_mkl_unary_hpp_

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>

namespace vsip
{
namespace impl
{
namespace mkl
{
template <template <typename> class Operator, typename Signature>
struct Unary { static bool const is_supported = false;};

// Operation, Function, Input type, Output type
#define UNARY(OP, FUN, IN, OUT)					\
void FUN(IN, OUT, length_type);				        \
template <>							\
struct Unary<vsip_csl::expr::op::OP, void(IN, OUT, length_type)>\
{								\
  static bool const is_supported = true;			\
  static char const *name() { return "mkl-unary-" #FUN;}	\
  static void exec(IN in, OUT out, length_type length)		\
  {								\
    FUN(in, out, length);					\
  }								\
};

UNARY(Sq, sq, float const *, float *);
UNARY(Sq, sq, double const *, double *);

UNARY(Conj, conj, complex<float> const *, complex<float> *);
UNARY(Conj, conj, complex<double> const *, complex<double> *);

UNARY(Mag, mag, float const *, float *);
UNARY(Mag, mag, double const *, double *);
UNARY(Mag, mag, complex<float> const *, float *);
UNARY(Mag, mag, complex<double> const *, double *);

UNARY(Arg, arg, complex<float> const *, float *);
UNARY(Arg, arg, complex<double> const *, double *);

UNARY(Sqrt, sqrt, float const *, float *);
UNARY(Sqrt, sqrt, double const *, double *);
UNARY(Sqrt, sqrt, complex<float> const *, complex<float> *);
UNARY(Sqrt, sqrt, complex<double> const *, complex<double> *);

UNARY(Exp, exp, float const *, float *);
UNARY(Exp, exp, double const *, double *);
UNARY(Exp, exp, complex<float> const *, complex<float> *);
UNARY(Exp, exp, complex<double> const *, complex<double> *);

UNARY(Log, log, float const *, float *);
UNARY(Log, log, double const *, double *);
UNARY(Log, log, complex<float> const *, complex<float> *);
UNARY(Log, log, complex<double> const *, complex<double> *);

UNARY(Log10, log10, float const *, float *);
UNARY(Log10, log10, double const *, double *);
UNARY(Log10, log10, complex<float> const *, complex<float> *);
UNARY(Log10, log10, complex<double> const *, complex<double> *);

UNARY(Sin, sin, float const *, float *);
UNARY(Sin, sin, double const *, double *);
UNARY(Sin, sin, complex<float> const *, complex<float> *);
UNARY(Sin, sin, complex<double> const *, complex<double> *);

UNARY(Cos, cos, float const *, float *);
UNARY(Cos, cos, double const *, double *);
UNARY(Cos, cos, complex<float> const *, complex<float> *);
UNARY(Cos, cos, complex<double> const *, complex<double> *);

UNARY(Tan, tan, float const *, float *);
UNARY(Tan, tan, double const *, double *);
UNARY(Tan, tan, complex<float> const *, complex<float> *);
UNARY(Tan, tan, complex<double> const *, complex<double> *);

UNARY(Acos, acos, float const *, float *);
UNARY(Acos, acos, double const *, double *);
UNARY(Acos, acos, complex<float> const *, complex<float> *);
UNARY(Acos, acos, complex<double> const *, complex<double> *);

UNARY(Asin, asin, float const *, float *);
UNARY(Asin, asin, double const *, double *);
UNARY(Asin, asin, complex<float> const *, complex<float> *);
UNARY(Asin, asin, complex<double> const *, complex<double> *);

UNARY(Atan, atan, float const *, float *);
UNARY(Atan, atan, double const *, double *);
UNARY(Atan, atan, complex<float> const *, complex<float> *);
UNARY(Atan, atan, complex<double> const *, complex<double> *);

UNARY(Sinh, sinh, float const *, float *);
UNARY(Sinh, sinh, double const *, double *);
UNARY(Sinh, sinh, complex<float> const *, complex<float> *);
UNARY(Sinh, sinh, complex<double> const *, complex<double> *);

UNARY(Cosh, cosh, float const *, float *);
UNARY(Cosh, cosh, double const *, double *);
UNARY(Cosh, cosh, complex<float> const *, complex<float> *);
UNARY(Cosh, cosh, complex<double> const *, complex<double> *);

UNARY(Tanh, tanh, float const *, float *);
UNARY(Tanh, tanh, double const *, double *);
UNARY(Tanh, tanh, complex<float> const *, complex<float> *);
UNARY(Tanh, tanh, complex<double> const *, complex<double> *);

// These are at present not part of the VSIPL++ spec:

// UNARY(Asinh, asinh, float const *, float *);
// UNARY(Asinh, asinh, double const *, double *);
// UNARY(Asinh, asinh, complex<float> const *, complex<float> *);
// UNARY(Asinh, asinh, complex<double> const *, complex<double> *);

// UNARY(Acosh, acosh, float const *, float *);
// UNARY(Acosh, acosh, double const *, double *);
// UNARY(Acosh, acosh, complex<float> const *, complex<float> *);
// UNARY(Acosh, acosh, complex<double> const *, complex<double> *);

// UNARY(Atanh, atanh, float const *, float *);
// UNARY(Atanh, atanh, double const *, double *);
// UNARY(Atanh, atanh, complex<float> const *, complex<float> *);
// UNARY(Atanh, atanh, complex<double> const *, complex<double> *);

#undef UNARY

} // namespace vsip::impl::mkl
} // namespace vsip::impl
} // namespace vsip

#endif

 
