/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/signal/corr_ext.hpp
    @author  Jules Bergmann
    @date    2005-10-05
    @brief   VSIPL++ Library: Correlation class implementation using Ext_data.
*/

#ifndef VSIP_OPT_SIGNAL_CORR_EXT_HPP
#define VSIP_OPT_SIGNAL_CORR_EXT_HPP

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
#include <vsip/core/signal/corr_common.hpp>
#include <vsip/core/extdata_dist.hpp>
#include <vsip/opt/dispatch.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
class Correlation
{
  static dimension_type const dim = D;

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
    output_size_(conv_output_size(Supp, ref_size_, input_size_, 1)),
    in_buffer_(input_size_.size()),
    out_buffer_(output_size_.size()),
    ref_buffer_(ref_size_.size())
  {}
  Correlation(Correlation const&) VSIP_NOTHROW;
  Correlation& operator=(Correlation const&) VSIP_NOTHROW;
  ~Correlation() VSIP_NOTHROW {}

  // Accessors.
public:
  Domain<dim> const& reference_size() const VSIP_NOTHROW  { return ref_size_; }
  Domain<dim> const& input_size() const VSIP_NOTHROW   { return input_size_; }
  Domain<dim> const& output_size() const VSIP_NOTHROW  { return output_size_; }

  float impl_performance(char* what) const
  {
    if (!strcmp(what, "in_ext_cost"))       return pm_in_ext_cost_;
    else if (!strcmp(what, "out_ext_cost")) return pm_out_ext_cost_;
    else return 0.f;
  }

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

  aligned_array<T> in_buffer_;
  aligned_array<T> out_buffer_;
  aligned_array<T> ref_buffer_;

  size_t          pm_ref_ext_cost_;
  size_t          pm_in_ext_cost_;
  size_t          pm_out_ext_cost_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Perform 1-D correlation.

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1,
	  typename Block2>
void
Correlation<D, Supp, T, n_times, a_hint>::impl_correlate(
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

  // PROFILE: Warn if arguments are not entirely on single processor
  // (either as undistributed views or as local views of distr obj).

  typedef typename Block_layout<Block0>::layout_type LP0;
  typedef typename Block_layout<Block1>::layout_type LP1;
  typedef typename Block_layout<Block2>::layout_type LP2;

  typedef Layout<1, Any_type, Any_type, Cmplx_inter_fmt> req_LP;

  typedef typename Adjust_layout<T, req_LP, LP0>::type use_LP0;
  typedef typename Adjust_layout<T, req_LP, LP1>::type use_LP1;
  typedef typename Adjust_layout<T, req_LP, LP2>::type use_LP2;

  typedef Ext_data_dist<Block0, SYNC_IN,  use_LP0> ref_ext_type;
  typedef Ext_data_dist<Block1, SYNC_IN,  use_LP1> in_ext_type;
  typedef Ext_data_dist<Block2, SYNC_OUT, use_LP2> out_ext_type;

  ref_ext_type ref_ext(ref.block(), ref_buffer_.get());
  in_ext_type  in_ext (in.block(),  in_buffer_.get());
  out_ext_type out_ext(out.block(), out_buffer_.get());

  pm_ref_ext_cost_ += ref_ext.cost();
  pm_in_ext_cost_  += in_ext.cost();
  pm_out_ext_cost_ += out_ext.cost();

  T* pref   = ref_ext.data();
  T* pin    = in_ext.data();
  T* pout   = out_ext.data();

  stride_type s_ref = ref_ext.stride(0);
  stride_type s_in  = in_ext.stride(0);
  stride_type s_out = out_ext.stride(0);

  if (Supp == support_full)
  {
    corr_full<T>(bias, pref, M, s_ref, pin, N, s_in, pout, P, s_out);
  }
  else if (Supp == support_same)
  {
    corr_same<T>(bias, pref, M, s_ref, pin, N, s_in, pout, P, s_out);
  }
  else // (Supp == support_min)
  {
    corr_min<T>(bias, pref, M, s_ref, pin, N, s_in, pout, P, s_out);
  }
}



// Perform 2-D correlation.

template <dimension_type      D,
	  support_region_type Supp,
	  typename            T,
	  unsigned            n_times,
          alg_hint_type       a_hint>
template <typename Block0,
	  typename Block1,
	  typename Block2>
void
Correlation<D, Supp, T, n_times, a_hint>::impl_correlate(
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

  // PROFILE: Warn if arguments are not entirely on single processor
  // (either as undistributed views or as local views of distr obj).

  typedef typename Block_layout<Block0>::layout_type LP0;
  typedef typename Block_layout<Block1>::layout_type LP1;
  typedef typename Block_layout<Block2>::layout_type LP2;

  typedef Layout<2, Any_type, Any_type, Cmplx_inter_fmt> req_LP;

  typedef typename Adjust_layout<T, req_LP, LP0>::type use_LP0;
  typedef typename Adjust_layout<T, req_LP, LP1>::type use_LP1;
  typedef typename Adjust_layout<T, req_LP, LP2>::type use_LP2;

  typedef Ext_data_dist<Block0, SYNC_IN,  use_LP0> ref_ext_type;
  typedef Ext_data_dist<Block1, SYNC_IN,  use_LP1> in_ext_type;
  typedef Ext_data_dist<Block2, SYNC_OUT, use_LP2> out_ext_type;

  ref_ext_type ref_ext(ref.block(), ref_buffer_.get());
  in_ext_type  in_ext (in.block(),  in_buffer_.get());
  out_ext_type out_ext(out.block(), out_buffer_.get());

  pm_ref_ext_cost_ += ref_ext.cost();
  pm_in_ext_cost_  += in_ext.cost();
  pm_out_ext_cost_ += out_ext.cost();

  T* p_ref   = ref_ext.data();
  T* p_in    = in_ext.data();
  T* p_out   = out_ext.data();

  stride_type ref_row_stride = ref_ext.stride(0);
  stride_type ref_col_stride = ref_ext.stride(1);
  stride_type in_row_stride  = in_ext.stride(0);
  stride_type in_col_stride  = in_ext.stride(1);
  stride_type out_row_stride = out_ext.stride(0);
  stride_type out_col_stride = out_ext.stride(1);

  if (Supp == support_full)
  {
    corr_full<T>(bias,
		 p_ref, Mr, Mc, ref_row_stride, ref_col_stride,
		 p_in, Nr, Nc, in_row_stride, in_col_stride,
		 p_out, Pr, Pc, out_row_stride, out_col_stride);
  }
  else if (Supp == support_same)
  {
    corr_same<T>(bias,
		 p_ref, Mr, Mc, ref_row_stride, ref_col_stride,
		 p_in, Nr, Nc, in_row_stride, in_col_stride,
		 p_out, Pr, Pc, out_row_stride, out_col_stride);
  }
  else // (Supp == support_min)
  {
    corr_min<T>(bias,
		p_ref, Mr, Mc, ref_row_stride, ref_col_stride,
		p_in, Nr, Nc, in_row_stride, in_col_stride,
		p_out, Pr, Pc, out_row_stride, out_col_stride);
  }
}
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <dimension_type      D,
          support_region_type R,
          typename            T,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<D, R, T, N, H>, be::generic>
{
  static bool const ct_valid = true;
  typedef vsip::impl::Correlation<D, R, T, N, H> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_SIGNAL_CORR_EXT_HPP
