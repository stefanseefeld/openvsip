/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA evaluators for ternary expressions.

#ifndef vsip_opt_cuda_ternary_hpp_
#define vsip_opt_cuda_ternary_hpp_

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/cuda/util.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{
template <template <typename, typename, typename> class Operator, typename Signature>
struct Ternary { static bool const is_supported = false;};

#define TERNARY(OP, FUNCTION, IN1, IN2, IN3, OUT)		\
template <>							\
struct Ternary<OP, void(IN1, IN2, IN3, OUT, length_type)>	\
{								\
  static bool const is_supported = true;			\
  static void exec(IN1 in1, IN2 in2, IN3 in3, OUT out, length_type length) \
  {								\
    FUNCTION(in1, in2, in3, out, length);			\
  }								\
};

void ma(float const *in1,
	float const *in2,
	float const *in3,
	float *output, length_type length);
void ma(std::complex<float> const *in1,
	std::complex<float> const *in2,
	std::complex<float> const *in3,
	std::complex<float> *output, length_type length);

TERNARY(vsip_csl::expr::op::Ma, ma, float const*, float const*, float const*, float*)
TERNARY(vsip_csl::expr::op::Ma, ma, complex<float> const*, complex<float> const*, complex<float> const*, complex<float>*)

void am(float const *in1,
	float const *in2,
	float const *in3,
	float *output, length_type length);
void am(std::complex<float> const *in1,
	std::complex<float> const *in2,
	std::complex<float> const *in3,
	std::complex<float> *output, length_type length);

TERNARY(vsip_csl::expr::op::Am, am, float const*, float const*, float const*, float*)
TERNARY(vsip_csl::expr::op::Am, am, complex<float> const*, complex<float> const*, complex<float> const*, complex<float>*)

#undef TERNARY

#define SIZE_THRESHOLD(OP, TYPE1, TYPE2, TYPE3, VAL)          \
template <>                                                   \
struct Size_threshold<expr::op::OP<TYPE1, TYPE2, TYPE3 > >    \
{                                                             \
  static length_type const value = VAL;                       \
};

// Experimentation has shown that the following thresholds
// work well for ma() in isolation, with input coming from
// the host and output going to the host.
SIZE_THRESHOLD(Ma,          float,          float,          float, 16384)
SIZE_THRESHOLD(Ma, complex<float>, complex<float>, complex<float>,  2048)

#undef SIZE_THRESHOLD

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

 
