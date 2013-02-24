/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/signal/conv_ext.hpp
    @author  Jules Bergmann
    @date    2005-06-09
    @brief   VSIPL++ Library: Convolution class implementation using dda::Data.
*/

#ifndef VSIP_OPT_SIGNAL_CONV_EXT_HPP
#define VSIP_OPT_SIGNAL_CONV_EXT_HPP

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
#include <vsip/dda.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/opt/dispatch.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

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
    coeff_ext_  (coeff_.block()),
    pcoeff_     (coeff_ext_.ptr()),
    kernel_size_(view_domain(coeff_)),
    input_size_ (input_size),
    output_size_(conv_output_size(Supp, kernel_size_, input_size,
                                  decimation)),
    in_buffer_(input_size_.size()),
    out_buffer_(output_size_.size()),
    tmp_buffer_(input_size_.size() + kernel_size_.size() - 1),
    decimation_ (decimation),
    pm_non_opt_calls_ (0)
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

  float impl_performance(char const *what) const
  {
    if (!strcmp(what, "in_ext_cost"))        return pm_in_ext_cost_;
    else if (!strcmp(what, "out_ext_cost"))  return pm_out_ext_cost_;
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

  typedef Layout<dim, typename Row_major<dim>::type,
                 unit_stride,
                 interleaved_complex>
    layout_type;
  typedef typename view_of<Dense<dim, T> >::type coeff_view_type;
  typedef dda::Data<typename coeff_view_type::block_type, dda::in, layout_type>
    c_ext_type;

  // Member data.
private:
  coeff_view_type coeff_;
  c_ext_type      coeff_ext_;
  T const *       pcoeff_;

  Domain<dim>     kernel_size_;
  Domain<dim>     input_size_;
  Domain<dim>     output_size_;

  aligned_array<T> in_buffer_;
  aligned_array<T> out_buffer_;
  aligned_array<T> tmp_buffer_;
  length_type     decimation_;

  int             pm_non_opt_calls_;
  size_t          pm_in_ext_cost_;
  size_t          pm_out_ext_cost_;
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
  using vsip::impl::Any_type;

  length_type const M = this->coeff_.size(0);
  length_type const N = this->input_size_[0].size();
  length_type const P = this->output_size_[0].size();

  assert(P == out.size());

  typedef typename get_block_layout<Block0>::type LP0;
  typedef typename get_block_layout<Block1>::type LP1;

  typedef Layout<1, Any_type, any_packing, interleaved_complex> req_LP;

  typedef typename adjust_layout<T, req_LP, LP0>::type use_LP0;
  typedef typename adjust_layout<T, req_LP, LP1>::type use_LP1;

  typedef dda::Data<Block0, dda::in, use_LP0>  in_ext_type;
  typedef dda::Data<Block1, dda::out, use_LP1> out_ext_type;

  in_ext_type  in_ext (in.block(), in_buffer_.get());
  out_ext_type out_ext(out.block(), out_buffer_.get());

  VSIP_IMPL_PROFILE(pm_in_ext_cost_  += in_ext.cost());
  VSIP_IMPL_PROFILE(pm_out_ext_cost_ += out_ext.cost());

  T const *pin = in_ext.ptr();
  T *pout = out_ext.ptr();

  stride_type s_in  = in_ext.stride(0);
  stride_type s_out = out_ext.stride(0);

  if (Supp == support_full)
  {
    conv_full<T>(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
  }
  else if (Supp == support_same)
  {
    conv_same<T>(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
  }
  else // (Supp == support_min)
  {
    conv_min<T>(pcoeff_, M, pin, N, s_in, pout, P, s_out, decimation_);
  }
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
  using vsip::impl::Any_type;

  // PROFILE: Warn if arguments are not entirely on single processor
  // (either as undistributed views or as local views of distr obj).

  length_type const Mr = this->coeff_.size(0);
  length_type const Mc = this->coeff_.size(1);

  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();

  length_type const Pr = this->output_size_[0].size();
  length_type const Pc = this->output_size_[1].size();

  assert(Pr == out.size(0) && Pc == out.size(1));

  typedef typename get_block_layout<Block0>::type LP0;
  typedef typename get_block_layout<Block1>::type LP1;

  typedef Layout<2, Any_type, any_packing, interleaved_complex> req_LP;

  typedef typename adjust_layout<T, req_LP, LP0>::type use_LP0;
  typedef typename adjust_layout<T, req_LP, LP1>::type use_LP1;

  typedef dda::Data<Block0, dda::in, use_LP0>  in_ext_type;
  typedef dda::Data<Block1, dda::out, use_LP1> out_ext_type;

  in_ext_type  in_ext (in.block(), in_buffer_.get());
  out_ext_type out_ext(out.block(), out_buffer_.get());

  VSIP_IMPL_PROFILE(pm_in_ext_cost_  += in_ext.cost());
  VSIP_IMPL_PROFILE(pm_out_ext_cost_ += out_ext.cost());

  T const *pin = in_ext.ptr();
  T *pout = out_ext.ptr();

  stride_type coeff_row_stride = coeff_ext_.stride(0);
  stride_type coeff_col_stride = coeff_ext_.stride(1);
  stride_type in_row_stride    = in_ext.stride(0);
  stride_type in_col_stride    = in_ext.stride(1);
  stride_type out_row_stride   = out_ext.stride(0);
  stride_type out_col_stride   = out_ext.stride(1);

  if (Supp == support_full)
  {
    conv_full<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		 pin, Nr, Nc, in_row_stride, in_col_stride,
		 pout, Pr, Pc, out_row_stride, out_col_stride,
		 decimation_);
  }
  else if (Supp == support_same)
  {
    conv_same<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		 pin, Nr, Nc, in_row_stride, in_col_stride,
		 pout, Pr, Pc, out_row_stride, out_col_stride,
		 decimation_);
  }
  else // (Supp == support_min)
  {
    conv_min<T>(pcoeff_, Mr, Mc, coeff_row_stride, coeff_col_stride,
		pin, Nr, Nc, in_row_stride, in_col_stride,
		pout, Pr, Pc, out_row_stride, out_col_stride,
		decimation_);
  }
}

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <symmetry_type       S,
	  support_region_type R,
          typename            T,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, T, N, H>, be::generic>
{
  static bool const ct_valid = true;
  typedef impl::Convolution<const_Vector, S, R, T, N, H> backend_type;
};
template <symmetry_type       S,
	  support_region_type R,
          typename            T,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, T, N, H>, be::generic>
{
  static bool const ct_valid = true;
  typedef impl::Convolution<const_Matrix, S, R, T, N, H> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_SIGNAL_CONV_EXT_HPP
