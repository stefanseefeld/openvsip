/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA evaluators for unary expressions.

#ifndef vsip_opt_cuda_unary_hpp_
#define vsip_opt_cuda_unary_hpp_

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/cuda/util.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{
template <template <typename> class Operator, typename Signature>
struct Unary { static bool const is_supported = false;};

#define UNARY(OP, FUNCTION, IN, OUT)				\
template <>							\
struct Unary<OP, void(IN, OUT, length_type)>			\
{								\
  static bool const is_supported = true;			\
  static void exec(IN in, OUT out, length_type length)		\
  {								\
    FUNCTION(in, out, length);					\
  }								\
};								\
								\
template <>							\
struct Unary<OP,						\
             void(IN, stride_type, OUT, stride_type,		\
                  length_type)>					\
{								\
  static bool const is_supported = true;			\
  static void exec(IN in, stride_type in_s,			\
                   OUT out, stride_type out_s,			\
                   length_type length)				\
  {								\
    FUNCTION(in, in_s, out, out_s, length);			\
  }								\
};								\
								\
template <>							\
struct Unary<OP,						\
             void(IN, stride_type, stride_type,			\
                  OUT, stride_type, stride_type,		\
                  length_type, length_type)>			\
{								\
  static bool const is_supported = true;			\
  static void exec(IN in, stride_type in_row_s,			\
                          stride_type in_col_s,			\
                   OUT out, stride_type out_row_s,		\
                            stride_type out_col_s,		\
                   length_type nrows, length_type ncols)	\
  {								\
    FUNCTION(in, in_row_s, in_col_s,				\
             out, out_row_s, out_col_s, nrows, ncols);		\
  }								\
};

void sqrt(float const *input, float *output, length_type length);
void sqrt(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void sqrt(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void sqrt(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void sqrt(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void sqrt(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void cos(float const *input, float *output, length_type length);
void cos(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void cos(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void cos(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void cos(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void cos(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void sin(float const *input, float *output, length_type length);
void sin(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void sin(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void sin(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void sin(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void sin(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void atan(float const *input, float *output, length_type length);
void atan(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void atan(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void log(float const *input, float *output, length_type length);
void log(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void log(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void log(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void log(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void log(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void log10(float const *input, float *output, length_type length);
void log10(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void log10(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void log10(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void log10(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void log10(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void acos(float const *input, float *output, length_type length);
void acos(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void acos(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void arg(std::complex<float> const *input, float *output,
  length_type length);
void arg(std::complex<float> const *input, stride_type in_stride,
  float *output, stride_type out_stride, length_type length);
void arg(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void asin(float const *input, float *output, length_type length);
void asin(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void asin(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void ceil(float const *input, float *output, length_type length);
void ceil(float const *input, stride_type in_stride, float *output,
  stride_type out_stride,  length_type length);
void ceil(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void ceil(std::complex<float> const *input, std::complex<float> *output, 
  length_type length);
void ceil(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void ceil(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void conj(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void conj(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void conj(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void cosh(float const *input, float *output, length_type length);
void cosh(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void cosh(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void cosh(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void cosh(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void cosh(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void euler(float const *input, std::complex<float> *output,
  length_type length);
void euler(float const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void euler(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void exp(float const *input, float *output, length_type length);
void exp(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void exp(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void exp(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void exp(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void exp(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void exp10(float const *input, float *output, length_type length);
void exp10(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void exp10(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void exp10(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void exp10(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void exp10(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void floor(float const *input, float *output, length_type length);
void floor(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void floor(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void floor(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void floor(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void floor(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void imag(std::complex<float> const *input, float *output, length_type length);
void imag(std::complex<float> const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void imag(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void mag(float const *input, float *output, length_type length);
void mag(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void mag(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void mag(std::complex<float> const *input, float *output,
  length_type length);
void mag(std::complex<float> const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void mag(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void magsq(float const *input, float *output, length_type length);
void magsq(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void magsq(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void magsq(std::complex<float> const *input, float *output,
  length_type length);
void magsq(std::complex<float> const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void magsq(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
 length_type num_rows, length_type num_cols);

void real(float const *input, float *output, length_type length);
void real(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void real(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void real(std::complex<float> const *input, float *output, length_type length);
void real(std::complex<float> const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void real(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void recip(float const *input, float *output, length_type length);
void recip(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void recip(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void recip(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void recip(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void recip(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void rsqrt(float const *input, float *output, length_type length);
void rsqrt(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void rsqrt(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void rsqrt(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void rsqrt(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void rsqrt(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void sinh(float const *input, float *output, length_type length);
void sinh(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void sinh(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void sinh(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void sinh(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void sinh(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void sq(float const *input, float *output, length_type length);
void sq(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void sq(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void sq(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void sq(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void sq(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void tan(float const *input, float *output, length_type length);
void tan(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void tan(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void tan(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void tan(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void tan(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);

void tanh(float const *input, float *output, length_type length);
void tanh(float const *input, stride_type in_stride, float *output,
  stride_type out_stride, length_type length);
void tanh(float const *input, stride_type row_in_stride, stride_type col_in_stride,
  float *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);
void tanh(std::complex<float> const *input, std::complex<float> *output,
  length_type length);
void tanh(std::complex<float> const *input, stride_type in_stride,
  std::complex<float> *output, stride_type out_stride, length_type length);
void tanh(std::complex<float> const *input, stride_type row_in_stride, stride_type col_in_stride,
  std::complex<float> *output, stride_type row_out_stride, stride_type col_out_stride,
  length_type num_rows, length_type num_cols);


UNARY(vsip_csl::expr::op::Sqrt, sqrt, float const*, float*)
UNARY(vsip_csl::expr::op::Sqrt, sqrt, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Cos, cos, float const*, float*)
UNARY(vsip_csl::expr::op::Cos, cos, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Sin, sin, float const*, float*)
UNARY(vsip_csl::expr::op::Sin, sin, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Atan, atan, float const*, float*)
UNARY(vsip_csl::expr::op::Log, log, float const*, float*)
UNARY(vsip_csl::expr::op::Log, log, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Log10, log10, float const*, float*)
UNARY(vsip_csl::expr::op::Log10, log10, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Acos, acos, float const*, float*)
UNARY(vsip_csl::expr::op::Arg, arg, complex<float> const*, float*)
UNARY(vsip_csl::expr::op::Asin, asin, float const*, float*)
UNARY(vsip_csl::expr::op::Ceil, ceil, float const*, float*)
UNARY(vsip_csl::expr::op::Ceil, ceil, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Conj, conj, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Cosh, cosh, float const*, float*)
UNARY(vsip_csl::expr::op::Cosh, cosh, complex<float>*, complex<float>*)
UNARY(vsip_csl::expr::op::Euler, euler, float const*, complex<float>*)
UNARY(vsip_csl::expr::op::Exp, exp, float const*, float*)
UNARY(vsip_csl::expr::op::Exp, exp, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Exp10, exp10, float const*, float*)
UNARY(vsip_csl::expr::op::Exp10, exp10, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Floor, floor, float const*, float*)
UNARY(vsip_csl::expr::op::Floor, floor, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Imag, imag, complex<float> const*, float*)
UNARY(vsip_csl::expr::op::Mag, mag, float const*, float*)
UNARY(vsip_csl::expr::op::Mag, mag, complex<float> const*, float*)
UNARY(vsip_csl::expr::op::Magsq, magsq, float const*, float*)
UNARY(vsip_csl::expr::op::Magsq, magsq, complex<float> const*, float*)
UNARY(vsip_csl::expr::op::Real, real, complex<float> const*, float*)
UNARY(vsip_csl::expr::op::Real, real, float const*, float*)
UNARY(vsip_csl::expr::op::Recip, recip, float const*, float*)
UNARY(vsip_csl::expr::op::Recip, recip, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Rsqrt, rsqrt, float const*, float*)
UNARY(vsip_csl::expr::op::Rsqrt, rsqrt, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Sinh, sinh, float const*, float*)
UNARY(vsip_csl::expr::op::Sinh, sinh, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Sq, sq, float const*, float*)
UNARY(vsip_csl::expr::op::Sq, sq, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Tan, tan, float const*, float*)
UNARY(vsip_csl::expr::op::Tan, tan, complex<float> const*, complex<float>*)
UNARY(vsip_csl::expr::op::Tanh, tanh, float const*, float*)
UNARY(vsip_csl::expr::op::Tanh, tanh, complex<float> const*, complex<float>*)


#undef UNARY

#define SIZE_THRESHOLD(OP, ATYPE, VAL)          \
template <>                                     \
struct Size_threshold<expr::op::OP<ATYPE > >    \
{                                               \
  static length_type const value = VAL;         \
};

// Experimentation has shown that the following thresholds
// work well in isolation, with input coming from
// the host and output going to the host.
//
SIZE_THRESHOLD(Sqrt,          float, 2048)
SIZE_THRESHOLD(Sqrt, complex<float>, 1024)
SIZE_THRESHOLD(Sin,           float, 2048)
SIZE_THRESHOLD(Sin,  complex<float>, 1024)
SIZE_THRESHOLD(Cos,           float, 2048)
SIZE_THRESHOLD(Cos,  complex<float>, 1024)
SIZE_THRESHOLD(Atan,          float, 2048)
SIZE_THRESHOLD(Log,           float, 2048)
SIZE_THRESHOLD(Log,  complex<float>, 1024)
SIZE_THRESHOLD(Log10,         float, 2048)
SIZE_THRESHOLD(Log10,complex<float>, 1024)
SIZE_THRESHOLD(Acos,          float, 2048)
SIZE_THRESHOLD(Asin,          float, 2048)
SIZE_THRESHOLD(Arg,  complex<float>, 1024)
SIZE_THRESHOLD(Ceil,          float, 2048)
SIZE_THRESHOLD(Ceil, complex<float>, 1024)
SIZE_THRESHOLD(Cosh,          float, 2048)
SIZE_THRESHOLD(Cosh, complex<float>, 1024)
SIZE_THRESHOLD(Conj, complex<float>, 1024)
SIZE_THRESHOLD(Euler,         float, 2048)
SIZE_THRESHOLD(Exp,           float, 2048)
SIZE_THRESHOLD(Exp,  complex<float>, 1024)
SIZE_THRESHOLD(Exp10,         float, 2048)
SIZE_THRESHOLD(Exp10,complex<float>, 1024)
SIZE_THRESHOLD(Floor,         float, 2048)
SIZE_THRESHOLD(Floor,complex<float>, 1024)
SIZE_THRESHOLD(Imag, complex<float>, 1024)
SIZE_THRESHOLD(Mag,           float, 2048)
SIZE_THRESHOLD(Mag,  complex<float>, 1024)
SIZE_THRESHOLD(Magsq,         float, 2048)
SIZE_THRESHOLD(Magsq,complex<float>, 1024)
SIZE_THRESHOLD(Real,          float, 2048)
SIZE_THRESHOLD(Real, complex<float>, 1024)
SIZE_THRESHOLD(Recip,         float, 2048)
SIZE_THRESHOLD(Recip,complex<float>, 1024)
SIZE_THRESHOLD(Rsqrt,         float, 2048)
SIZE_THRESHOLD(Rsqrt,complex<float>, 1024)
SIZE_THRESHOLD(Sinh,          float, 2048)
SIZE_THRESHOLD(Sinh, complex<float>, 1024)
SIZE_THRESHOLD(Sq,            float, 2048)
SIZE_THRESHOLD(Sq,   complex<float>, 1024)
SIZE_THRESHOLD(Tan,           float, 2048)
SIZE_THRESHOLD(Tan,  complex<float>, 1024)
SIZE_THRESHOLD(Tanh,          float, 2048)
SIZE_THRESHOLD(Tanh, complex<float>, 1024)

#undef SIZE_THRESHOLD

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

 
