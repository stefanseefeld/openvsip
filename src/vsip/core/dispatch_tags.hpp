/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/dispatch_tags.hpp
    @author  Stefan Seefeld
    @date    2009-07-03
    @brief   VSIPL++ Library: dispatch tags.

*/

#ifndef VSIP_CORE_DISPATCH_TAGS_HPP
#define VSIP_CORE_DISPATCH_TAGS_HPP

namespace vsip_csl
{
namespace dispatcher
{
/// backend tags
namespace be
{
/// Hook for custom evaluators
struct user;
/// Intel IPP Library
struct intel_ipp;
/// Optimized Matrix Transpose
struct transpose;
/// Mercury SAL Library
struct mercury_sal;
/// IBM CBE SDK.
struct cbe_sdk;
/// IBM Cell Math Library
struct cml;
/// Builtin SIMD routines (non loop fusion)
struct simd_builtin;
/// Dense multi-dim expr reduction
struct dense_expr;
/// Optimized Copy
struct copy;
/// Special expr handling (vmmul, etc)
struct op_expr;
/// SIMD Loop Fusion.
struct simd_loop_fusion;
struct simd_unaligned_loop_fusion;
/// Fused Fastconv RBO evaluator.
struct fc_expr;
/// Return-block expression evaluator.
struct rbo_expr;
/// Multi-dim expr reduction
struct mdim_expr;
/// Generic Loop Fusion (base case).
struct loop_fusion;
/// FFTW3.
struct fftw3;
/// Dummy FFT
struct no_fft;

/// BLAS implementation (ATLAS, MKL, etc)
struct blas;
/// LAPACK implementation (ATLAS, MKL, etc)
struct lapack;
/// Generic implementation.
struct generic;
/// Parallel implementation.
struct parallel;
/// C-VSIPL library.
struct cvsip;
/// NVidia CUDA GPU library
struct cuda;
/// Optimized Tag.struct 
struct opt;

} // namespace vsip_csl::dispatcher::be

/// Operation tags
namespace op
{
template <dimension_type D,        /// dimension
	  typename I,              /// input type
	  typename O,              /// output type
	  int S,                   /// special dimension
	  return_mechanism_type R, /// return-mechanism type
	  unsigned N>              /// number of times
struct fft;
template <typename I,              /// input type
	  typename O,              /// output type
	  int A,                   /// axis
	  int D,                   /// direction
	  return_mechanism_type R, /// return-mechanism type
	  unsigned N>              /// number of times
struct fftm;
struct fir;
struct freqswap;
struct hist;
/// dot-product
struct dot;
/// outer-product
struct outer;
/// matrix product
struct prod;
/// conjugate matrix product
struct prodj;
/// generalized matrix-matrix product
struct gemp;
/// lower-upper linear system solver
struct lud;
/// QR decomposition
struct qrd;
/// Cholesky solver
struct chold;
/// singular value decomposition
struct svd;

} // namespace vsip_csl::dispatcher::op
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl


#endif
