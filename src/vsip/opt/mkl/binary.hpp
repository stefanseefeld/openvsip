/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   MKL bindings for binary expressions.

#ifndef vsip_opt_mkl_binary_hpp_
#define vsip_opt_mkl_binary_hpp_

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>

namespace vsip
{
namespace impl
{
namespace mkl
{
// Operation, Function, Input1 type, Input2 type, Output type
template <template <typename, typename> class Operator, typename Signature>
struct Binary { static bool const is_supported = false;};

#define BINARY(OP, FUN, IN1, IN2, OUT)				\
void FUN(IN1, IN2, OUT, length_type);				\
template <>							\
struct Binary<vsip_csl::expr::op::OP, void(IN1, IN2, OUT, length_type)>	\
{								\
  static bool const is_supported = true;			\
  static char const *name() { return "mkl-binary-" #FUN;}	\
  static void exec(IN1 in1, IN2 in2, OUT out, length_type length)	\
  {								\
    FUN(in1, in2, out, length);					\
  }								\
};

BINARY(Add, add, float const *, float const *, float *);
BINARY(Add, add, double const *, double const *, double *);
BINARY(Add, add, complex<float> const *, complex<float> const *, complex<float> *);
BINARY(Add, add, complex<double> const *, complex<double> const *, complex<double> *);

BINARY(Sub, sub, float const *, float const *, float *);
BINARY(Sub, sub, double const *, double const *, double *);
BINARY(Sub, sub, complex<float> const *, complex<float> const *, complex<float> *);
BINARY(Sub, sub, complex<double> const *, complex<double> const *, complex<double> *);

BINARY(Mult, mul, float const *, float const *, float *);
BINARY(Mult, mul, double const *, double const *, double *);
BINARY(Mult, mul, complex<float> const *, complex<float> const *, complex<float> *);
BINARY(Mult, mul, complex<double> const *, complex<double> const *, complex<double> *);

BINARY(Div, div, float const *, float const *, float *);
BINARY(Div, div, double const *, double const *, double *);
BINARY(Div, div, complex<float> const *, complex<float> const *, complex<float> *);
BINARY(Div, div, complex<double> const *, complex<double> const *, complex<double> *);

BINARY(Pow, pow, float const *, float const *, float *);
BINARY(Pow, pow, double const *, double const *, double *);
BINARY(Pow, pow, complex<float> const *, complex<float> const *, complex<float> *);
BINARY(Pow, pow, complex<double> const *, complex<double> const *, complex<double> *);

BINARY(Atan2, atan2, float const *, float const *, float *);
BINARY(Atan2, atan2, double const *, double const *, double *);

BINARY(Hypot, hypot, float const *, float const *, float *);
BINARY(Hypot, hypot, double const *, double const *, double *);

#undef BINARY

} // namespace vsip::impl::mkl
} // namespace vsip::impl
} // namespace vsip

#endif

 
