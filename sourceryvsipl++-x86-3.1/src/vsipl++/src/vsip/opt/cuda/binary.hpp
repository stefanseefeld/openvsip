/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA evaluators for binary expressions.

#ifndef vsip_opt_cuda_binary_hpp_
#define vsip_opt_cuda_binary_hpp_

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/cuda/util.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{
template <template <typename, typename> class Operator, typename Signature>
struct Binary { static bool const is_supported = false;};

#define BINARY(OP, FUNCTION, IN1, IN2, OUT)			\
template <>							\
struct Binary<OP, void(IN1, IN2, OUT, length_type)>		\
{								\
  static bool const is_supported = true;			\
  static void exec(IN1 in1, IN2 in2, OUT out, length_type length) \
  {								\
    FUNCTION(in1, in2, out, length);				\
  }								\
};								\
								\
template <>							\
struct Binary<OP,						\
	      void(IN1, stride_type, IN2, stride_type,		\
		   OUT, stride_type, length_type)>		\
{								\
  static bool const is_supported = true;			\
  static void exec(IN1 in1, stride_type in1_s,                  \
                   IN2 in2, stride_type in2_s,                  \
                   OUT out, stride_type out_s,			\
		   length_type length)				\
  {								\
    FUNCTION(in1, in1_s, in2, in2_s, out, out_s, length);	\
  }								\
};								\
								\
template <>							\
struct Binary<OP,						\
	      void(IN1, stride_type, stride_type,               \
                   IN2, stride_type, stride_type,       	\
		   OUT, stride_type, stride_type,		\
                   length_type, length_type)>	        	\
{								\
  static bool const is_supported = true;			\
  static void exec(IN1 in1, stride_type in1_row_s,		\
                            stride_type in1_col_s,              \
                   IN2 in2, stride_type in2_row_s,              \
			    stride_type in2_col_s,		\
                   OUT out, stride_type out_row_s,		\
			    stride_type out_col_s,		\
		   length_type nrows, length_type ncols)	\
  {								\
    FUNCTION(in1, in1_row_s, in1_col_s,				\
             in2, in2_row_s, in2_col_s,				\
	     out, out_row_s, out_col_s, nrows, ncols);		\
  }								\
};

void vadd(float const *in1,
	  float const *in2,
          float *output, length_type length);
void vadd(std::complex<float> const *in1,
	  std::complex<float> const *in2,
          std::complex<float> *output, length_type length);
void vadd(float const *in1, stride_type in1_stride,
	  float const *in2, stride_type in2_stride,
          float *output, stride_type out_stride, length_type length);
void vadd(std::complex<float> const *in1, stride_type in1_stride,
	  std::complex<float> const *in2, stride_type in2_stride,
          std::complex<float> *output, stride_type out_stride, length_type length);
void vadd(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          float *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);
void vadd(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Add, vadd, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Add, vadd, complex<float> const*, complex<float> const*, complex<float>*)

void vsub(float const *in1,
	  float const *in2,
          float *output, length_type length);
void vsub(std::complex<float> const *in1,
	  std::complex<float> const *in2,
          std::complex<float> *output, length_type length);
void vsub(float const *in1, stride_type in1_stride,
	  float const *in2, stride_type in2_stride,
          float *output, stride_type out_stride, length_type length);
void vsub(std::complex<float> const *in1, stride_type in1_stride,
	  std::complex<float> const *in2, stride_type in2_stride,
          std::complex<float> *output, stride_type out_stride, length_type length);
void vsub(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          float *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);
void vsub(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Sub, vsub, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Sub, vsub, complex<float> const*, complex<float> const*, complex<float>*)

void vmul(float const *in1,
	  float const *in2,
          float *output, length_type length);
void vmul(std::complex<float> const *in1,
	  std::complex<float> const *in2,
          std::complex<float> *output, length_type length);
void vmul(float const *in1, stride_type in1_stride,
	  float const *in2, stride_type in2_stride,
          float *output, stride_type out_stride, length_type length);
void vmul(std::complex<float> const *in1, stride_type in1_stride,
	  std::complex<float> const *in2, stride_type in2_stride,
          std::complex<float> *output, stride_type out_stride, length_type length);
void vmul(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          float *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);
void vmul(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Mult, vmul, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Mult, vmul, complex<float> const*, complex<float> const*, complex<float>*)

void vdiv(float const *in1,
	  float const *in2,
          float *output, length_type length);
void vdiv(std::complex<float> const *in1,
	  std::complex<float> const *in2,
          std::complex<float> *output, length_type length);
void vdiv(float const *in1, stride_type in1_stride,
	  float const *in2, stride_type in2_stride,
          float *output, stride_type out_stride, length_type length);
void vdiv(std::complex<float> const *in1, stride_type in1_stride,
	  std::complex<float> const *in2, stride_type in2_stride,
          std::complex<float> *output, stride_type out_stride, length_type length);
void vdiv(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          float *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);
void vdiv(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Div, vdiv, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Div, vdiv, complex<float> const*, complex<float> const*, complex<float>*)

void atan2(float const *in1,
	   float const *in2,
           float *output, length_type length);
void atan2(float const *in1, stride_type in1_stride,
	   float const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void atan2(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	   float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Atan2, atan2, float const*, float const*, float*)

void fmod(float const *in1,
          float const *in2,
          float *output, length_type length);
void fmod(float const *in1, stride_type in1_stride,
          float const *in2, stride_type in2_stride,
          float *output, stride_type out_stride, length_type length);
void fmod(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          float *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Fmod, fmod, float const*, float const*, float*)

void hypot(float const *in1,
           float const *in2,
           float *output, length_type length);
void hypot(float const *in1, stride_type in1_stride,
           float const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void hypot(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	   float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Hypot, hypot, float const*, float const*, float*)

void jmul(std::complex<float> const *in1,
          std::complex<float> const *in2,
          std::complex<float> *output, length_type length);
void jmul(float const               *in1,
          std::complex<float> const *in2,
          std::complex<float> *output, length_type length);
void jmul(std::complex<float> const *in1, stride_type in1_stride,
          std::complex<float> const *in2, stride_type in2_stride,
          std::complex<float> *output, stride_type out_stride, length_type length);
void jmul(float const               *in1, stride_type in1_stride,
          std::complex<float> const *in2, stride_type in2_stride,
          std::complex<float> *output, stride_type out_stride, length_type length);
void jmul(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	  std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);
void jmul(float const               *in1, stride_type in1_row_stride, stride_type in1_col_stride,
          std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
          std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
          length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Jmul, jmul, complex<float> const*, complex<float> const*, complex<float>*)
BINARY(vsip_csl::expr::op::Jmul, jmul, float const*, complex<float> const*, complex<float>*)

void max(float const *in1,
         float const *in2,
         float *output, length_type length);
void max(float const *in1, stride_type in1_stride,
         float const *in2, stride_type in2_stride,
         float *output, stride_type out_stride, length_type length);
void max(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	 float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
         float *output, stride_type out_row_stride, stride_type out_col_stride,
         length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Max, max, float const*, float const*, float*)

void maxmg(float const *in1,
           float const *in2,
           float *output, length_type length);
void maxmg(std::complex<float> const *in1,
           std::complex<float> const *in2,
           float *output, length_type length);
void maxmg(std::complex<float> const *in1,
           float const *in2,
           float *output, length_type length);
void maxmg(float const *in1,
           std::complex<float> const *in2,
           float *output, length_type length);
void maxmg(float const *in1, stride_type in1_stride,
           float const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void maxmg(std::complex<float> const *in1, stride_type in1_stride,
           std::complex<float> const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void maxmg(std::complex<float> const *in1, stride_type in1_stride,
           float const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void maxmg(float const *in1, stride_type in1_stride,
           std::complex<float> const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void maxmg(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	   float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);
void maxmg(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	   std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);
void maxmg(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
           float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);
void maxmg(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
           std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Maxmg, maxmg, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Maxmg, maxmg, complex<float> const*, complex<float> const*, float*)
BINARY(vsip_csl::expr::op::Maxmg, maxmg, complex<float> const*, float const*, float*)
BINARY(vsip_csl::expr::op::Maxmg, maxmg, float const*, complex<float> const*, float*)

void maxmgsq(float const *in1,
             float const *in2,
             float *output, length_type length);
void maxmgsq(std::complex<float> const *in1,
             std::complex<float> const *in2,
             float *output, length_type length);
void maxmgsq(std::complex<float> const *in1,
             float const *in2,
             float *output, length_type length);
void maxmgsq(float const *in1,
             std::complex<float> const *in2,
             float *output, length_type length);
void maxmgsq(float const *in1, stride_type in1_stride,
             float const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void maxmgsq(std::complex<float> const *in1, stride_type in1_stride,
             std::complex<float> const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void maxmgsq(std::complex<float> const *in1, stride_type in1_stride,
             float const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void maxmgsq(float const *in1, stride_type in1_stride,
             std::complex<float> const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void maxmgsq(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	     float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);
void maxmgsq(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	     std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);
void maxmgsq(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
             float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);
void maxmgsq(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
             std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Maxmgsq, maxmgsq, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Maxmgsq, maxmgsq, complex<float> const*, complex<float> const*, float*)
BINARY(vsip_csl::expr::op::Maxmgsq, maxmgsq, complex<float> const*, float const*, float*)
BINARY(vsip_csl::expr::op::Maxmgsq, maxmgsq, float const*, complex<float> const*, float*)

void min(float const *in1,
         float const *in2,
         float *output, length_type length);
void min(float const *in1, stride_type in1_stride,
         float const *in2, stride_type in12_stride,
         float *output, stride_type out_stride, length_type length);
void min(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	 float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
         float *output, stride_type out_row_stride, stride_type out_col_stride,
         length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Min, min, float const*, float const*, float*)

void minmg(float const *in1,
           float const *in2,
           float *output, length_type length);
void minmg(std::complex<float> const *in1,
           std::complex<float> const *in2,
           float *output, length_type length);
void minmg(std::complex<float> const *in1,
           float const *in2,
           float *output, length_type length);
void minmg(float const *in1,
           std::complex<float> const *in2,
           float *output, length_type length);
void minmg(float const *in1, stride_type in1_stride,
           float const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void minmg(std::complex<float> const *in1, stride_type in1_stride,
           std::complex<float> const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void minmg(std::complex<float> const *in1, stride_type in1_stride,
           float const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void minmg(float const *in1, stride_type in1_stride,
           std::complex<float> const *in2, stride_type in2_stride,
           float *output, stride_type out_stride, length_type length);
void minmg(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	   float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);
void minmg(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	   std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);
void minmg(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
           float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);
void minmg(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
           std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
           float *output, stride_type out_row_stride, stride_type out_col_stride,
           length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Minmg, minmg, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Minmg, minmg, complex<float> const*, complex<float> const*, float*)
BINARY(vsip_csl::expr::op::Minmg, minmg, complex<float> const*, float const*, float*)
BINARY(vsip_csl::expr::op::Minmg, minmg, float const*, complex<float> const*, float*)

void minmgsq(float const *in1,
             float const *in2,
             float *output, length_type length);
void minmgsq(std::complex<float> const *in1,
             std::complex<float> const *in2,
             float *output, length_type length);
void minmgsq(std::complex<float> const *in1,
             float const *in2,
             float *output, length_type length);
void minmgsq(float const *in1,
             std::complex<float> const *in2,
             float *output, length_type length);
void minmgsq(float const *in1, stride_type in1_stride,
             float const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void minmgsq(std::complex<float> const *in1, stride_type in1_stride,
             std::complex<float> const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void minmgsq(std::complex<float> const *in1, stride_type in1_stride,
             float const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void minmgsq(float const *in1, stride_type in1_stride,
             std::complex<float> const *in2, stride_type in2_stride,
             float *output, stride_type out_stride, length_type length);
void minmgsq(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	     float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);
void minmgsq(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	     std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);
void minmgsq(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
             float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);
void minmgsq(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
             std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
             float *output, stride_type out_row_stride, stride_type out_col_stride,
             length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Minmgsq, minmgsq, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Minmgsq, minmgsq, complex<float> const*, complex<float> const*, float*)
BINARY(vsip_csl::expr::op::Minmgsq, minmgsq, complex<float> const*, float const*, float*)
BINARY(vsip_csl::expr::op::Minmgsq, minmgsq, float const*, complex<float> const*, float*)

void pow(float const *in1,
         float const *in2,
         float *output, length_type length);
void pow(std::complex<float> const *in1,
         float const *in2,
         std::complex<float> *output, length_type length);
void pow(float const *in1,
         std::complex<float> const *in2,
         std::complex<float> *output, length_type length);
void pow(std::complex<float> const *in1,
         std::complex<float> const *in2,
         std::complex<float> *output, length_type length);
void pow(float const *in1, stride_type in1_stride,
         float const *in2, stride_type in2_stride,
         float *output, stride_type out_stride, length_type length);
void pow(std::complex<float> const *in1, stride_type in1_stride,
         float const *in2, stride_type in2_stride,
         std::complex<float> *output, stride_type out_stride, length_type length);
void pow(float const *in1, stride_type in1_stride,
         std::complex<float> const *in2, stride_type in2_stride,
         std::complex<float> *output, stride_type out_stride, length_type length);
void pow(std::complex<float> const *in1, stride_type in1_stride,
         std::complex<float> const *in2, stride_type in2_stride,
         std::complex<float> *output, stride_type out_stride, length_type length);
void pow(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	 float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
         float *output, stride_type out_row_stride, stride_type out_col_stride,
         length_type num_rows, length_type num_cols);
void pow(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
         float const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
         std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
         length_type num_rows, length_type num_cols);
void pow(float const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
         std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
         std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
         length_type num_rows, length_type num_cols);
void pow(std::complex<float> const *in1, stride_type in1_row_stride, stride_type in1_col_stride,
	 std::complex<float> const *in2, stride_type in2_row_stride, stride_type in2_col_stride,
         std::complex<float> *output, stride_type out_row_stride, stride_type out_col_stride,
         length_type num_rows, length_type num_cols);

BINARY(vsip_csl::expr::op::Pow, pow, float const*, float const*, float*)
BINARY(vsip_csl::expr::op::Pow, pow, complex<float> const*, float const*, complex<float>*)
BINARY(vsip_csl::expr::op::Pow, pow, float const*, complex<float> const*, complex<float>*)
BINARY(vsip_csl::expr::op::Pow, pow, complex<float> const*, complex<float> const*, complex<float>*)

#undef BINARY

#define SIZE_THRESHOLD(OP, LTYPE, RTYPE, VAL)          \
template <>                                            \
struct Size_threshold<expr::op::OP<LTYPE, RTYPE > >    \
{                                                      \
  static length_type const value = VAL;                \
};

// These numbers were chosen to maximize performance while taking into account
//  the cost of data transfers before and after the operation.  Sub, Mult, 
//  and Add use higher thresholds than the other functions listed.  This is
//  because testing has indicated these other functions (Div, Max, Min, Fmod, etc)
//  may maintain a larger performance advantage over the CPU counterparts.
//  This may be due to the availability of CUDA intrinsic functions for the 
//  underlying operations which give CUDA an advantage over the CPU.
//
SIZE_THRESHOLD(Atan2,               float,               float, 2048)
SIZE_THRESHOLD(Sub,                 float,               float, 65536)
SIZE_THRESHOLD(Sub,   std::complex<float>, std::complex<float>, 32768)
SIZE_THRESHOLD(Mult,                float,               float, 65536)
SIZE_THRESHOLD(Mult,  std::complex<float>, std::complex<float>, 32768)
SIZE_THRESHOLD(Add,                 float,               float, 65536)
SIZE_THRESHOLD(Add,  std::complex<float>, std::complex<float>,  32768)
SIZE_THRESHOLD(Div,                 float,               float, 2048)
SIZE_THRESHOLD(Div,  std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Max,                 float,               float, 2048)
SIZE_THRESHOLD(Min,                 float,               float, 2048)
SIZE_THRESHOLD(Fmod,                float,               float, 2048)
SIZE_THRESHOLD(Pow,                 float,               float, 2048)
SIZE_THRESHOLD(Pow,  std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Pow,  std::complex<float>,               float,  1024)
SIZE_THRESHOLD(Pow,                float, std::complex<float>,  2048)
SIZE_THRESHOLD(Minmgsq,             float,               float, 2048)
SIZE_THRESHOLD(Minmgsq,std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Minmgsq,std::complex<float>,               float,  1024)
SIZE_THRESHOLD(Minmgsq,            float, std::complex<float>,  2048)
SIZE_THRESHOLD(Minmg,               float,               float, 2048)
SIZE_THRESHOLD(Minmg,std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Minmg,std::complex<float>,               float,  1024)
SIZE_THRESHOLD(Minmg,              float, std::complex<float>,  2048)
SIZE_THRESHOLD(Maxmgsq,             float,               float, 2048)
SIZE_THRESHOLD(Maxmgsq,std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Maxmgsq,std::complex<float>,               float,  1024)
SIZE_THRESHOLD(Maxmgsq,            float, std::complex<float>,  2048)
SIZE_THRESHOLD(Maxmg,               float,               float, 2048)
SIZE_THRESHOLD(Maxmg,std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Maxmg,std::complex<float>,               float,  1024)
SIZE_THRESHOLD(Maxmg,              float, std::complex<float>,  2048)
SIZE_THRESHOLD(Jmul,               float, std::complex<float>,  2048)
SIZE_THRESHOLD(Jmul, std::complex<float>, std::complex<float>,  1024)
SIZE_THRESHOLD(Hypot,               float,               float, 2048)

#undef SIZE_THRESHOLD

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif

 
