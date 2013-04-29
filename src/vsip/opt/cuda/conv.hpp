/* Copyright (c) 2005, 2006, 2007, 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Convolution class implementation using CUDA

#ifndef VSIP_OPT_CUDA_CONV_HPP
#define VSIP_OPT_CUDA_CONV_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>

#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/dispatch.hpp>


namespace vsip
{
namespace impl
{
namespace cuda
{

// 1-D convolution for real input/coefficients -> real output
extern void
conv(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  length_type         decimation,
  length_type         shift,
  bool                isconv,
  bool                even,
  bool                same,
  bool                min,
  int                 bias);

extern void
conv(
  std::complex<float> const* input,
  std::complex<float> const* coeff,
  std::complex<float>*       output,
  length_type                input_length,
  length_type                coeff_length,
  length_type                output_length,
  length_type                decimation,
  length_type                shift,
  bool                       isconv,
  bool                       even,
  bool                       same,
  bool                       min,
  int                        bias);

// 2-D convolution for real input/coefficients -> real output
extern void
conv_2d(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_decimation,
  length_type         col_decimation,
  stride_scalar_type  row_stride,
  stride_type         row_shift,
  stride_type         col_shift,
  bool                isconv,
  int                 bias);

extern void
conv_2d(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_decimation,
  length_type                       col_decimation,
  stride_scalar_type                row_stride,
  stride_type                       row_shift,
  stride_type                       col_shift,
  bool                              isconv,
  int                               bias);

extern void
conv_no_decimation_min(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length);

extern void
conv_no_decimation_min(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length);

extern void
conv_no_decimation_full(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length);

extern void
conv_no_decimation_full(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length);

extern void
conv_no_decimation_same_odd(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  length_type         shift);

extern void
conv_no_decimation_same_odd(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length,
  length_type                       shift);

extern void
conv_no_decimation_same_even(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  length_type         shift);

extern void
conv_no_decimation_same_even(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length,
  length_type                       shift);

extern void
conv_2d_no_decimation_min(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_shift,
  length_type         col_shift);

extern void
conv_2d_no_decimation_min(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_shift,
  length_type                       col_shift);

extern void
conv_2d_no_decimation_full(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_shift,
  length_type         col_shift);

extern void
conv_2d_no_decimation_full(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_shift,
  length_type                       col_shift);

extern void
conv_2d_no_decimation_same_nrow_even_ncol_even(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_shift,
  length_type         col_shift);

extern void
conv_2d_no_decimation_same_nrow_even_ncol_even(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_shift,
  length_type                       col_shift);

extern void
conv_2d_no_decimation_same_nrow_even_ncol_odd(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_shift,
  length_type         col_shift);

extern void
conv_2d_no_decimation_same_nrow_even_ncol_odd(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_shift,
  length_type                       col_shift);

extern void
conv_2d_no_decimation_same_nrow_odd_ncol_even(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_shift,
  length_type         col_shift);

extern void
conv_2d_no_decimation_same_nrow_odd_ncol_even(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_shift,
  length_type                       col_shift);

extern void
conv_2d_no_decimation_same_nrow_odd_ncol_odd(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_num_rows,
  length_type         input_num_cols,
  length_type         coeff_num_rows,
  length_type         coeff_num_cols,
  length_type         output_num_rows,
  length_type         output_num_cols,
  length_type         row_shift,
  length_type         col_shift);

extern void
conv_2d_no_decimation_same_nrow_odd_ncol_odd(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_num_rows,
  length_type                       input_num_cols,
  length_type                       coeff_num_rows,
  length_type                       coeff_num_cols,
  length_type                       output_num_rows,
  length_type                       output_num_cols,
  length_type                       row_shift,
  length_type                       col_shift);


template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
class Convolution
{
  static dimension_type const dim = Dim_of_view<ConstViewT>::dim;

  // Compile-time constants.
public:
  static symmetry_type const       symmtry = Symm;
  static support_region_type const supprt  = Supp;

  // Constructors, copies, assignments, and destructors.
public:
  template <typename Block>
  Convolution(ConstViewT<T, Block> filter_coeffs,
              Domain<dim> const&   input_size,
              length_type          decimation)
    VSIP_THROW((std::bad_alloc))
  : coeff_      (conv_kernel<coeff_view_type>(Symm, filter_coeffs)),
    dev_coeff_  (coeff_.block()),
    pcoeff_     (dev_coeff_.ptr()),
    kernel_size_(view_domain(coeff_)),
    input_size_ (input_size),
    output_size_(conv_output_size(Supp, kernel_size_, input_size,
                                  decimation)),
    decimation_ (decimation)
  {}
  Convolution(Convolution const&) VSIP_NOTHROW;
  Convolution& operator=(Convolution const&) VSIP_NOTHROW;
  ~Convolution() VSIP_NOTHROW {}

  // Accessors.
public:
  Domain<dim> const& kernel_size() const VSIP_NOTHROW  { return kernel_size_; }
  Domain<dim> const& filter_order() const VSIP_NOTHROW { return kernel_size_; }
  Domain<dim> const& input_size() const VSIP_NOTHROW   { return input_size_; }
  Domain<dim> const& output_size() const VSIP_NOTHROW  { return output_size_; }
  symmetry_type symmetry() const VSIP_NOTHROW          { return Symm; }
  support_region_type support() const VSIP_NOTHROW     { return Supp; }
  length_type decimation() const VSIP_NOTHROW          { return decimation_; }

  // Implementation functions.
protected:
  template <typename Block0,
	    typename Block1>
  void
  convolve(const_Vector<T, Block0>,
	   Vector<T, Block1>)
    VSIP_NOTHROW;

  template <typename Block0,
	    typename Block1>
  void
  convolve(const_Matrix<T, Block0>,
	   Matrix<T, Block1>)
    VSIP_NOTHROW;

  typedef typename view_of<Dense<dim, T> >::type coeff_view_type;
  typedef dda::Data<typename coeff_view_type::block_type, dda::in> coeff_dev_type;

  // Member data.
private:
  coeff_view_type coeff_;
  coeff_dev_type  dev_coeff_;
  T const *       pcoeff_;

  Domain<dim>     kernel_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;

  length_type     decimation_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Perform 1-D convolution.

template <template <typename, typename> class ConstViewT,
	  symmetry_type       Symm,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1>
void
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::
convolve(
  const_Vector<T, Block0> in,
  Vector<T, Block1>       out)
VSIP_NOTHROW
{
  length_type const M = this->coeff_.size(0);
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();
  assert(P == out.size());
  assert(N == in.size());

  dda::Data<Block0, dda::in> dev_in(in.block());
  dda::Data<Block1, dda::out> dev_out(out.block());

  T const* pin    = dev_in.ptr();
  T* pout         = dev_out.ptr();

  length_type shift = 0;

  // Device properties
  unsigned long const shared_mem_size_bytes = 16384;
  unsigned long const max_threads_per_block = 512;

  bool is_odd = (M % 2 == 1);
  if (Supp == support_same)
  {
    shift = M / 2;
  }
  else if (Supp == support_min)
  {
    shift = M - 1;
  }

  // The decimation must be one to use the specialized kernels.  The coefficient
  //  length is limited by the size of shared memory on the device.  40 is the 
  //  size of the input parameters to each kernel which also takes up shared
  //  memory.
  if (decimation_ == 1 && M <= vsip::min((shared_mem_size_bytes - 40) / sizeof(T) / 4,
                                    max_threads_per_block))
  {
    if (Supp == support_min)
      conv_no_decimation_min(pin, pcoeff_, pout, N, M, P);
    else if (Supp == support_full)
      conv_no_decimation_full(pin, pcoeff_, pout, N, M, P);
    else if (Supp == support_same && is_odd)
      conv_no_decimation_same_odd(pin, pcoeff_, pout, N, M, P, shift);
    else
      conv_no_decimation_same_even(pin, pcoeff_, pout, N, M, P, shift);
  }
  else
    conv(pin, pcoeff_, pout, N, M, P, decimation_, shift, true, false, false, false, 0);
}



// Perform 2-D convolution.

template <template <typename, typename> class ConstViewT,
	  symmetry_type          Symm,
	  support_region_type    Supp,
	  typename               T,
	  unsigned               n_times,
          alg_hint_type          a_hint>
template <typename Block0,
	  typename Block1>
void
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::
convolve(
  const_Matrix<T, Block0> in,
  Matrix<T, Block1>       out)
VSIP_NOTHROW
{
  length_type const Mr = this->coeff_.size(0);
  length_type const Mc = this->coeff_.size(1);
  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();
  length_type const Pr = this->output_size_[0].size();
  length_type const Pc = this->output_size_[1].size();
  assert(Pr == out.size(0) && Pc == out.size(1));
  assert(Nr == in.size(0) && Nc == in.size(1));

  dda::Data<Block0, dda::in> dev_in(in.block());
  dda::Data<Block1, dda::out> dev_out(out.block());
  assert(dev_in.stride(1) == 1); // Verify that the along-row stride is one

  T const* pin = dev_in.ptr();
  T* pout = dev_out.ptr();

  // Device properties
  unsigned long const shared_mem_size_bytes = 16384;
  unsigned long const max_threads_per_block_x = 32;
  unsigned long const max_threads_per_block_y = 16;

  length_type optimization_threshold_Mr = vsip::min(int(max_threads_per_block_x),
          int(std::sqrt(float(shared_mem_size_bytes - 72) / (sizeof(T) * 2.5))));

  length_type optimization_threshold_Mc = optimization_threshold_Mr / 2;

  stride_type in_row_stride    = dev_in.stride(0);
  stride_type out_row_stride   = dev_out.stride(0);

  stride_type shiftr = 0;
  stride_type shiftc = 0;

  bool is_nrows_even = (Mr % 2 == 0);
  bool is_ncols_even = (Mc % 2 == 0);
  if (Supp == support_same)
  {
    shiftr = Mr / 2;
    shiftc = Mc / 2;
  }
  else if (Supp == support_min)
  {
    shiftr = Mr - 1;
    shiftc = Mc - 1;
  }

  // The decimation must be one to use the specialized kernels.  The coefficient
  //  lengths are limited by the size of shared memory on the device.  72 is the 
  //  size of the input parameter stack to each kernel which also takes up shared
  //  memory.
  if (decimation_ == 1 && Mr <= optimization_threshold_Mr &&
                          Mc <= optimization_threshold_Mc)
  {
    if (Supp == support_min)
      conv_2d_no_decimation_min(pin, pcoeff_, pout, Nr, Nc, Mr, Mc, Pr, Pc,
                                                             shiftr, shiftc);
    else if (Supp == support_full)
      conv_2d_no_decimation_full(pin, pcoeff_, pout, Nr, Nc, Mr, Mc, Pr, Pc,
                                                             shiftr, shiftc);
    else if (Supp == support_same && is_ncols_even && is_nrows_even)
      conv_2d_no_decimation_same_nrow_even_ncol_even(pin, pcoeff_, pout, Nr,
                                         Nc, Mr, Mc, Pr, Pc, shiftr, shiftc);
    else if (Supp == support_same && !is_ncols_even && is_nrows_even)
      conv_2d_no_decimation_same_nrow_even_ncol_odd(pin, pcoeff_, pout, Nr,
                                         Nc, Mr, Mc, Pr, Pc, shiftr, shiftc);
    else if (Supp == support_same && is_ncols_even && !is_nrows_even)
      conv_2d_no_decimation_same_nrow_odd_ncol_even(pin, pcoeff_, pout, Nr,
                                         Nc, Mr, Mc, Pr, Pc, shiftr, shiftc);
    else
      conv_2d_no_decimation_same_nrow_odd_ncol_odd(pin, pcoeff_, pout, Nr, Nc,
                                             Mr, Mc, Pr, Pc, shiftr, shiftc);
  }
  else
    conv_2d(pin, pcoeff_, pout, Nr, Nc, Mr, Mc, Pr, Pc, decimation_, decimation_,
          dev_in.stride(0), shiftr, shiftc, true, 0);
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

// Enable 1-D convolution for float
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, float, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Convolution<const_Vector, S, R, float, N, H> backend_type;
};

// Enable 2-D convolution for float
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, float, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Convolution<const_Matrix, S, R, float, N, H> backend_type;
};

// Enable 1-D convolution for std::complex<float>
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, std::complex<float>, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Convolution<const_Vector, S, R, std::complex<float>,
                                  N, H> backend_type;
};

// Enable 2-D convolution for std::complex<float>
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, std::complex<float>, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Convolution<const_Matrix, S, R, std::complex<float>,
                                  N, H> backend_type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_CUDA_CONV_HPP
