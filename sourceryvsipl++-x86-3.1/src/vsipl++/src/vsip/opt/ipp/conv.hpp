/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/conv.hpp
    @author  Jules Bergmann
    @date    2005-08-31
    @brief   VSIPL++ Library: Convolution class implementation using IPP.
*/

#ifndef VSIP_IMPL_SIGNAL_CONV_IPP_HPP
#define VSIP_IMPL_SIGNAL_CONV_IPP_HPP

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
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/opt/ipp/bindings.hpp>
#include <vsip/opt/dispatch.hpp>
/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace ipp
{
template <template <typename, typename> class V,
	  symmetry_type       S,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Convolution
{
  static dimension_type const dim = Dim_of_view<V>::dim;

  // Compile-time constants.
public:
  static symmetry_type const       symmtry = S;
  static support_region_type const supprt  = R;

  // Constructors, copies, assignments, and destructors.
public:
  template <typename Block>
  Convolution(V<T, Block> filter_coeffs,
              Domain<dim> const& input_size,
              length_type decimation)
    VSIP_THROW((std::bad_alloc));

  Convolution(Convolution const&) VSIP_NOTHROW;
  Convolution& operator=(Convolution const&) VSIP_NOTHROW;
  ~Convolution() VSIP_NOTHROW;

  // Accessors.
public:
  Domain<dim> const& kernel_size() const VSIP_NOTHROW  { return kernel_size_; }
  Domain<dim> const& filter_order() const VSIP_NOTHROW { return kernel_size_; }
  Domain<dim> const& input_size() const VSIP_NOTHROW   { return input_size_; }
  Domain<dim> const& output_size() const VSIP_NOTHROW  { return output_size_; }
  symmetry_type symmetry() const VSIP_NOTHROW          { return symmtry; }
  support_region_type support() const VSIP_NOTHROW     { return supprt; }
  length_type decimation() const VSIP_NOTHROW          { return decimation_; }

  float impl_performance(char* what) const
  {
    if (!strcmp(what, "in_dda_cost")) return pm_in_dda_cost_;
    else if (!strcmp(what, "out_dda_cost")) return pm_out_dda_cost_;
    else if (!strcmp(what, "non-opt-calls")) return pm_non_opt_calls_;
    else return 0.f;
  }

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

  typedef Layout<dim, typename Row_major<dim>::type, unit_stride, interleaved_complex>
    layout_type;
  typedef typename view_of<Dense<dim, T> >::type coeff_view_type;
  typedef dda::Data<typename coeff_view_type::block_type, dda::in, layout_type>
    c_data_type;

private:
  coeff_view_type coeff_;
  c_data_type     coeff_data_;
  T const *       pcoeff_;

  Domain<dim>     kernel_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;
  T*              in_buffer_;
  T*              out_buffer_;
  T*              tmp_buffer_;
  length_type     decimation_;

  int             pm_non_opt_calls_;
  size_t          pm_in_dda_cost_;
  size_t          pm_out_dda_cost_;
};

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
template <typename Block>
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::Convolution(
  ConstViewT<T, Block> filter_coeffs,
  Domain<dim> const&   input_size,
  length_type          decimation)
VSIP_THROW((std::bad_alloc))
  : coeff_      (conv_kernel<coeff_view_type>(Symm, filter_coeffs)),
    coeff_data_ (coeff_.block()),
    pcoeff_     (coeff_data_.ptr()),
    kernel_size_(impl::view_domain(coeff_)),
    input_size_ (input_size),
    output_size_(impl::conv_output_size(Supp, kernel_size_, input_size,
					decimation)),
    decimation_ (decimation),
    pm_non_opt_calls_ (0)
{
  in_buffer_  = new T[input_size_.size()];
  if (in_buffer_ == NULL)
    VSIP_IMPL_THROW(std::bad_alloc());

  out_buffer_ = new T[output_size_.size()];
  if (out_buffer_ == NULL)
  {
    delete[] in_buffer_;
    VSIP_IMPL_THROW(std::bad_alloc());
  }

  tmp_buffer_ = new T[input_size.size() + kernel_size_.size() - 1];
  if (tmp_buffer_ == NULL)
  {
    delete[] out_buffer_;
    delete[] in_buffer_;
    VSIP_IMPL_THROW(std::bad_alloc());
  }
}

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::~Convolution()
  VSIP_NOTHROW
{
  delete[] tmp_buffer_;
  delete[] out_buffer_;
  delete[] in_buffer_;
}



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
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::convolve(
  const_Vector<T, Block0> in,
  Vector<T, Block1>       out)
VSIP_NOTHROW
{
  length_type const M = this->coeff_.size(0);
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  assert(P == out.size());

  typedef vsip::dda::Data<Block0, dda::in> in_data_type;
  typedef vsip::dda::Data<Block1, dda::out> out_data_type;

  in_data_type  in_data(in.block(),  in_buffer_);
  out_data_type out_data(out.block(), out_buffer_);

  VSIP_IMPL_PROFILE(pm_in_dda_cost_  += in_data.cost());
  VSIP_IMPL_PROFILE(pm_out_dda_cost_ += out_data.cost());

  T const *pin    = in_data.ptr();
  T *pout   = out_data.ptr();

  stride_type s_in  = in_data.stride(0);
  stride_type s_out = out_data.stride(0);

  if (Supp == support_full)
  {
    if (decimation_ == 1 && in_data.stride(0) == 1 && out_data.stride(0) == 1)
    {
      impl::ipp::conv(pcoeff_, M, pin, N, pout);
    }
    else
    {
      VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
      conv_full<T>(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
    }
  }
  else if (Supp == support_same)
  {
    if (decimation_ == 1 && in_data.stride(0) == 1)
    {
      impl::ipp::conv(pcoeff_, M, pin, N, tmp_buffer_);
      for (index_type n=0; n<P; ++n)
	pout[n * s_out] = tmp_buffer_[n+M/2];
    }
    else
    {
      VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
      conv_same<T>(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
    }
  }
  else // (Supp == support_min)
  {
    if (decimation_ == 1 && in_data.stride(0) == 1)
    {
      impl::ipp::conv(pcoeff_, M, pin, N, tmp_buffer_);
      for (index_type n=0; n<P; ++n)
	pout[n * s_out] = tmp_buffer_[n+M-1];
    }
    else
    {
      VSIP_IMPL_PROFILE(pm_non_opt_calls_++);
      conv_min<T>(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
    }
  }
}



// Perform 2-D convolution.

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T,
	  unsigned                            n_times,
          alg_hint_type                       a_hint>
template <typename Block0,
	  typename Block1>
void
Convolution<ConstViewT, Symm, Supp, T, n_times, a_hint>::convolve(
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

  typedef vsip::dda::Data<Block0, dda::in> in_data_type;
  typedef vsip::dda::Data<Block1, dda::out> out_data_type;

  in_data_type in_data(in.block(), in_buffer_);
  out_data_type out_data(out.block(), out_buffer_);

  VSIP_IMPL_PROFILE(pm_in_dda_cost_  += in_data.cost());
  VSIP_IMPL_PROFILE(pm_out_dda_cost_ += out_data.cost());

  T const *pin    = in_data.ptr();
  T *pout   = out_data.ptr();

  stride_type coeff_row_stride = coeff_data_.stride(0);
  stride_type coeff_col_stride = coeff_data_.stride(1);
  stride_type in_row_stride    = in_data.stride(0);
  stride_type in_col_stride    = in_data.stride(1);
  stride_type out_row_stride   = out_data.stride(0);
  stride_type out_col_stride   = out_data.stride(1);

  if (Supp == support_full)
  {
    if (decimation_ == 1 && coeff_col_stride == 1 &&
	in_data.stride(1) == 1 && out_data.stride(1) == 1)
    {
      impl::ipp::conv_full_2d(
	pcoeff_, Mr, Mc, coeff_row_stride,
	pin, Nr, Nc, in_row_stride,
	pout, out_row_stride);
    }
    else
    {
      conv_full<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		   pin, Nr, Nc, in_row_stride, in_col_stride,
		   pout, Pr, Pc, out_row_stride, out_col_stride,
		   decimation_);
    }
  }
  else if (Supp == support_same)
  {
    if (decimation_ == 1 && coeff_col_stride == 1 &&
	in_data.stride(1) == 1 && out_data.stride(1) == 1)
    {
      // IPP only provides full- and min-support convolutions.
      // We implement same-support by doing a min-support and
      // then filling out the edges.

      index_type n0_r = (Mr - 1) - (Mr/2);
      index_type n0_c = (Mc - 1) - (Mc/2);
      index_type n1_r = Nr - (Mr/2);
      index_type n1_c = Nc - (Mc/2);

      T* pout_adj = pout + (n0_r)*out_row_stride
			 + (n0_c)*out_col_stride;

      if (n1_r > n0_r && n1_c > n0_c)
	impl::ipp::conv_valid_2d(
	  pcoeff_, Mr, Mc, coeff_row_stride,
	  pin, Nr, Nc, in_row_stride,
	  pout_adj, out_row_stride);

      conv_same_edge<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		   pin, Nr, Nc, in_row_stride, in_col_stride,
		   pout, Pr, Pc, out_row_stride, out_col_stride,
		   decimation_);
    }
    else
    {
      conv_same<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		   pin, Nr, Nc, in_row_stride, in_col_stride,
		   pout, Pr, Pc, out_row_stride, out_col_stride,
		   decimation_);
    }
  }
  else // (Supp == support_min)
  {
    if (decimation_ == 1 && coeff_col_stride == 1 &&
	in_data.stride(1) == 1 && out_data.stride(1) == 1)
    {
      impl::ipp::conv_valid_2d(
	pcoeff_, Mr, Mc, coeff_row_stride,
	pin, Nr, Nc, in_row_stride,
	pout, out_row_stride);
    }
    else
    {
      conv_min<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		  pin, Nr, Nc, in_row_stride, in_col_stride,
		  pout, Pr, Pc, out_row_stride, out_col_stride,
		  decimation_);
    }
  }
}
} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, float, N, H>, be::intel_ipp>
{
  static bool const ct_valid = true;
  typedef impl::ipp::Convolution<const_Vector, S, R, float, N, H> backend_type;
};
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, double, N, H>, be::intel_ipp>
{
  static bool const ct_valid = true;
  typedef impl::ipp::Convolution<const_Vector, S, R, double, N, H> backend_type;
};
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, short, N, H>, be::intel_ipp>
{
  static bool const ct_valid = true;
  typedef impl::ipp::Convolution<const_Matrix, S, R, short, N, H> backend_type;
};
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, float, N, H>, be::intel_ipp>
{
  static bool const ct_valid = true;
  typedef impl::ipp::Convolution<const_Matrix, S, R, float, N, H> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_SIGNAL_CONV_IPP_HPP
