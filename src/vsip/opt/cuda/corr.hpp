/* Copyright (c) 2005, 2006, 2007, 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   Correlation class implementation using CUDA


#ifndef VSIP_OPT_CUDA_CORR_HPP
#define VSIP_OPT_CUDA_CORR_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/dispatch.hpp>

/***********************************************************************
  Declarations
***********************************************************************/




namespace vsip
{
namespace impl
{
namespace cuda
{

extern void
corr_no_decimation_min(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  int                 bias);

extern void
corr_no_decimation_min(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length,
  int                               bias);

extern void
corr_no_decimation_full(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  int                 bias);

extern void
corr_no_decimation_full(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length,
  int                               bias);

extern void
corr_no_decimation_same_odd(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  length_type         shift,
  int                 bias);

extern void
corr_no_decimation_same_odd(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length,
  length_type                       shift,
  int                               bias);

extern void
corr_no_decimation_same_even(
  float const*        input,
  float const*        coeff,
  float*              output,
  length_type         input_length,
  length_type         coeff_length,
  length_type         output_length,
  length_type         shift,
  int                 bias);

extern void
corr_no_decimation_same_even(
  std::complex<float> const*        input,
  std::complex<float> const*        coeff,
  std::complex<float>*              output,
  length_type                       input_length,
  length_type                       coeff_length,
  length_type                       output_length,
  length_type                       shift,
  int                               bias);

extern void
corr_2d_no_decimation_min(
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
  length_type         col_shift,
  int                 bias);

extern void
corr_2d_no_decimation_min(
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
  length_type                       col_shift,
  int                               bias);

extern void
corr_2d_no_decimation_full(
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
  length_type         col_shift,
  int                 bias);

extern void
corr_2d_no_decimation_full(
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
  length_type                       col_shift,
  int                               bias);

extern void
corr_2d_no_decimation_same_nrow_even_ncol_even(
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
  length_type         col_shift,
  int                 bias);

extern void
corr_2d_no_decimation_same_nrow_even_ncol_even(
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
  length_type                       col_shift,
  int                               bias);

extern void
corr_2d_no_decimation_same_nrow_even_ncol_odd(
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
  length_type         col_shift,
  int                 bias);

extern void
corr_2d_no_decimation_same_nrow_even_ncol_odd(
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
  length_type                       col_shift,
  int                               bias);

extern void
corr_2d_no_decimation_same_nrow_odd_ncol_even(
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
  length_type         col_shift,
  int                 bias);

extern void
corr_2d_no_decimation_same_nrow_odd_ncol_even(
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
  length_type                       col_shift,
  int                               bias);

extern void
corr_2d_no_decimation_same_nrow_odd_ncol_odd(
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
  length_type         col_shift,
  int                 bias);

extern void
corr_2d_no_decimation_same_nrow_odd_ncol_odd(
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
  length_type                       col_shift,
  int                               bias);


template <template <typename, typename> class ConstViewT,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
class Correlation
{
  static dimension_type const dim = Dim_of_view<ConstViewT>::dim;

  // Compile-time constants.
public:
  static support_region_type const supprt  = Supp;

  // Constructors, copies, assignments, and destructors.
public:
  Correlation(Domain<dim> const&   ref_size,
              Domain<dim> const&   input_size)
    VSIP_THROW((std::bad_alloc))
  : ref_size_   (normalize(ref_size)),
    input_size_ (normalize(input_size)),
    output_size_(conv_output_size(Supp, ref_size_, input_size_, 1))
  {}
  Correlation(Correlation const&) VSIP_NOTHROW;
  Correlation& operator=(Correlation const&) VSIP_NOTHROW;
  ~Correlation() VSIP_NOTHROW {}

  // Accessors.
public:
  Domain<dim> const& reference_size() const VSIP_NOTHROW  { return ref_size_; }
  Domain<dim> const& input_size() const VSIP_NOTHROW   { return input_size_; }
  Domain<dim> const& output_size() const VSIP_NOTHROW  { return output_size_; }

  // Implementation functions.
public:
  template <typename Block0,
	    typename Block1,
	    typename Block2>
  void
  impl_correlate(bias_type bias,
                 const_Vector<T, Block0> ref,
                 const_Vector<T, Block1> in,
                 Vector<T, Block2>       out)
    VSIP_NOTHROW;

  template <typename Block0,
	    typename Block1,
	    typename Block2>
  void
  impl_correlate(bias_type bias,
                 const_Matrix<T, Block0> ref,
                 const_Matrix<T, Block1> in,
                 Matrix<T, Block2>       out)
    VSIP_NOTHROW;

  // Member data.
private:
  Domain<dim>     ref_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Perform 1-D correlation.

template <template <typename, typename> class ConstViewT,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1,
	  typename Block2>
void
Correlation<ConstViewT, Supp, T, n_times, a_hint>::impl_correlate(
  bias_type               bias,
  const_Vector<T, Block0> ref,
  const_Vector<T, Block1> in,
  Vector<T, Block2>       out)
VSIP_NOTHROW
{
  length_type const M = this->ref_size_[0].size();
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  assert(M == ref.size());
  assert(N == in.size());
  assert(P == out.size());

  dda::Data<Block0, dda::in> dev_ref(ref.block());
  dda::Data<Block1, dda::in> dev_in(in.block());
  dda::Data<Block2, dda::out> dev_out(out.block());

  T const* pref   = dev_ref.ptr();
  T const* pin    = dev_in.ptr();
  T* pout         = dev_out.ptr();

  length_type shift = 0;

  // Device properties
  unsigned long const shared_mem_size_bytes = 16384;
  unsigned long const max_threads_per_block = 512;

  bool is_even = (M % 2 == 0);
  bool is_same = false;
  bool is_min = false;

  if (Supp == support_same)
  {
    shift = M / 2;
    is_same = true;
  }
  else if (Supp == support_min)
  {
    shift = M - 1;
    is_min = true;
  }

  // The decimation must be one to use the specialized kernels.  The coefficient
  //  length is limited by the size of shared memory on the device.  44 is the 
  //  size of the input parameters to each kernel which also takes up shared
  //  memory.
  if (M <= vsip::min((shared_mem_size_bytes - 44) / sizeof(T) / 4,
                max_threads_per_block) && (Supp == support_min || Supp == support_full))
  {
    if (Supp == support_min)
      corr_no_decimation_min(pin, pref, pout, N, M, P, int(bias));
    else if (Supp == support_full)
      corr_no_decimation_full(pin, pref, pout, N, M, P, int(bias));
  }
  else if (M <= vsip::min((shared_mem_size_bytes - 44) / sizeof(T) / 4,
                max_threads_per_block) && Supp == support_same && bias == biased)
  {
    if (is_even)
      corr_no_decimation_same_even(pin, pref, pout, N, M, P, shift, int(bias));
    else
      corr_no_decimation_same_odd(pin, pref, pout, N, M, P, shift, int(bias));
  }
  else
    conv(pin, pref, pout, N, M, P, 1, shift, false, is_even, is_same, is_min, int(bias));
}



// Perform 2-D correlation.

template <template <typename, typename> class ConstViewT,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1,
	  typename Block2>
void
Correlation<ConstViewT, Supp, T, n_times, a_hint>::impl_correlate(
  bias_type               bias,
  const_Matrix<T, Block0> ref,
  const_Matrix<T, Block1> in,
  Matrix<T, Block2>       out)
VSIP_NOTHROW
{
  using vsip::impl::Any_type;

  length_type const Mr = this->ref_size_[0].size();
  length_type const Mc = this->ref_size_[1].size();
  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();
  length_type const Pr = this->output_size_[0].size();
  length_type const Pc = this->output_size_[1].size();

  assert(Mr == ref.size(0));
  assert(Mc == ref.size(1));
  assert(Nr == in.size(0));
  assert(Nc == in.size(1));
  assert(Pr == out.size(0));
  assert(Pc == out.size(1));

  dda::Data<Block0, dda::in> dev_ref(ref.block());
  dda::Data<Block1, dda::in> dev_in(in.block());
  dda::Data<Block2, dda::out> dev_out(out.block());

  T const* pref   = dev_ref.ptr();
  T const* pin    = dev_in.ptr();
  T* pout         = dev_out.ptr();

  // Device properties
  unsigned long const shared_mem_size_bytes = 16384;
  unsigned long const max_threads_per_block_x = 32;
  unsigned long const max_threads_per_block_y = 16;

  length_type optimization_threshold_Mr = vsip::min(int(max_threads_per_block_x),
          int(std::sqrt(float(shared_mem_size_bytes - 76) / (sizeof(T) * 2.5))));

  length_type optimization_threshold_Mc = optimization_threshold_Mr / 2;

  stride_type shiftr = -(Mr - 1);
  stride_type shiftc = -(Mc - 1);

  bool is_nrows_even = (Mr % 2 == 0);
  bool is_ncols_even = (Mc % 2 == 0);
  if (Supp == support_same)
  {
    shiftr = -(Mr / 2);
    shiftc = -(Mc / 2);

  }
  else if (Supp == support_min)
  {
    shiftr = 0;
    shiftc = 0;

  }
  // The decimation must be one to use the specialized kernels and outputs for
  //  same support must be biased.  The coefficient lengths are limited by the
  //  size of shared memory on the device.  76 is the size of the input
  //  parameter stack to each kernel which also takes up shared memory.
  if (Mr <= optimization_threshold_Mr && Mc <= optimization_threshold_Mc &&
     (Supp == support_min || Supp == support_full ||
     (Supp == support_same && bias == biased)))
  {
    if (Supp == support_min)
      corr_2d_no_decimation_min(pin, pref, pout, Nr, Nc, Mr, Mc, Pr, Pc,
                                                   shiftr, shiftc, int(bias));
    else if (Supp == support_full)
      corr_2d_no_decimation_full(pin, pref, pout, Nr, Nc, Mr, Mc, Pr, Pc,
                                                   shiftr, shiftc, int(bias));

    else if (Supp == support_same && is_ncols_even && is_nrows_even)
      corr_2d_no_decimation_same_nrow_even_ncol_even(pin, pref, pout, Nr,
                               Nc, Mr, Mc, Pr, Pc, shiftr, shiftc, int(bias));

    else if (Supp == support_same && !is_ncols_even && is_nrows_even)
      corr_2d_no_decimation_same_nrow_even_ncol_odd(pin, pref, pout, Nr,
                               Nc, Mr, Mc, Pr, Pc, shiftr, shiftc, int(bias));

    else if (Supp == support_same && is_ncols_even && !is_nrows_even)
      corr_2d_no_decimation_same_nrow_odd_ncol_even(pin, pref, pout, Nr,
                               Nc, Mr, Mc, Pr, Pc, shiftr, shiftc, int(bias));
    else
      corr_2d_no_decimation_same_nrow_odd_ncol_odd(pin, pref, pout, Nr, Nc,
                                   Mr, Mc, Pr, Pc, shiftr, shiftc, int(bias));
  }
  else
    conv_2d(pin, pref, pout, Nr, Nc, Mr, Mc, Pr, Pc, 1, 1, dev_in.stride(0),
          shiftr, shiftc, false, int(bias));


}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

// Enable 1-D correlation for float
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, float, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Correlation<const_Vector, R, float, N, H> backend_type;
};

// Enable 2-D correlation for float
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<2, R, float, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Correlation<const_Matrix, R, float, N, H> backend_type;
};

// Enable 1-D correlation for std::complex<float>
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, std::complex<float>, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Correlation<const_Vector, R, std::complex<float>,
                                  N, H> backend_type;
};

// Enable 2-D correlation for std::complex<float>
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<2, R, std::complex<float>, N, H>, be::cuda>
{
  static bool const ct_valid = true;
  typedef impl::cuda::Correlation<const_Matrix, R, std::complex<float>,
                                  N, H> backend_type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_CORR_HPP
