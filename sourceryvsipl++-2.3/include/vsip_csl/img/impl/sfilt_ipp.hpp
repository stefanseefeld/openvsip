/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_CSL_IMG_IMPL_SFILT_IPP_HPP
#define VSIP_CSL_IMG_IMPL_SFILT_IPP_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/core/extdata_dist.hpp>
#include <vsip_csl/img/impl/sfilt_common.hpp>
#include <ippi.h>

namespace vsip_csl
{
namespace img
{
namespace impl
{
namespace ipp
{
/// Specialize Sfilt for using ext data.
template <vsip::support_region_type R,
	  edge_handling_type E>
class Sfilt
{
  static vsip::dimension_type const dim = 2;

public:
  static vsip::support_region_type const support_tv = R;
  static edge_handling_type        const edge_tv    = E;

  template <typename Block1, typename Block2>
  Sfilt(vsip::const_Vector<float, Block1>  coeff0,	// coeffs for dimension 0
	vsip::const_Vector<float, Block2>  coeff1,	// coeffs for dimension 1
	vsip::Domain<dim> const &input_size)
    VSIP_THROW((std::bad_alloc));

  ~Sfilt() VSIP_NOTHROW;

  vsip::Domain<dim> const& kernel_size() const VSIP_NOTHROW
  { return kernel_size_;}
  vsip::Domain<dim> const& filter_order() const VSIP_NOTHROW
  { return kernel_size_;}
  vsip::Domain<dim> const& input_size() const VSIP_NOTHROW
  { return input_size_;}
  vsip::Domain<dim> const& output_size() const VSIP_NOTHROW
  { return output_size_;}
  vsip::support_region_type support() const VSIP_NOTHROW
  { return R;}

  float impl_performance(char const *what) const
  {
    if (!strcmp(what, "in_ext_cost"))        return pm_in_ext_cost_;
    else if (!strcmp(what, "out_ext_cost"))  return pm_out_ext_cost_;
    else if (!strcmp(what, "non-opt-calls")) return pm_non_opt_calls_;
    else return 0.f;
  }

  template <typename Block0, typename Block1>
  void
  filter(vsip::const_Matrix<float, Block0>, vsip::Matrix<float, Block1>) VSIP_NOTHROW;

  typedef vsip::impl::Layout<1,
			     typename vsip::impl::Row_major<dim>::type,
			     vsip::impl::Stride_unit,
			     vsip::impl::Cmplx_inter_fmt>
		layout_type;
  typedef typename vsip::impl::View_of_dim<1, float, vsip::Dense<1, float> >::type
		coeff_view_type;
  typedef vsip::impl::Ext_data<typename coeff_view_type::block_type,
			       layout_type>
		c_ext_type;

private:
  Sfilt(Sfilt const&) VSIP_NOTHROW;
  Sfilt &operator=(Sfilt const&) VSIP_NOTHROW;

  coeff_view_type coeff0_;
  coeff_view_type coeff1_;
  c_ext_type      coeff0_ext_;
  c_ext_type      coeff1_ext_;
  float *pcoeff0_;
  float *pcoeff1_;

  vsip::Domain<dim>     kernel_size_;
  vsip::Domain<dim>     input_size_;
  vsip::Domain<dim>     output_size_;

  float *in_buffer_;
  float *out_buffer_;
  float *tmp_buffer_;

  int             pm_non_opt_calls_;
  size_t          pm_in_ext_cost_;
  size_t          pm_out_ext_cost_;
};

/// Construct a convolution object.
template <vsip::support_region_type R,
	  edge_handling_type E>
template <typename Block1, typename Block2>
Sfilt<R, E>::Sfilt(vsip::const_Vector<float, Block1> coeff0, // coeffs for dimension 0
		   vsip::const_Vector<float, Block2> coeff1, // coeffs for dimension 1
		   vsip::Domain<dim> const &input_size)
  VSIP_THROW((std::bad_alloc))
  : coeff0_     (coeff0.size()),
    coeff1_     (coeff1.size()),
    coeff0_ext_ (coeff0_.block(), vsip::impl::SYNC_IN),
    coeff1_ext_ (coeff1_.block(), vsip::impl::SYNC_IN),
    pcoeff0_    (coeff0_ext_.data()),
    pcoeff1_    (coeff1_ext_.data()),

    kernel_size_(vsip::Domain<2>(coeff0.size(), coeff1.size())),
    input_size_ (input_size),
    output_size_(sfilt_output_size(R, kernel_size_, input_size)),
    pm_non_opt_calls_ (0)
{
  coeff0_ = coeff0(vsip::Domain<1>(coeff0.length()-1, -1, coeff0.length()));
  coeff1_ = coeff1(vsip::Domain<1>(coeff1.length()-1, -1, coeff1.length()));

  in_buffer_  = new float[input_size_.size()];
  out_buffer_ = new float[output_size_.size()];
  tmp_buffer_ = new float[output_size_.size()];
}

/// Destroy a generic Convolution_impl object.
template <vsip::support_region_type R,
	  edge_handling_type E>
Sfilt<R, E>::~Sfilt() VSIP_NOTHROW
{
  delete[] tmp_buffer_;
  delete[] out_buffer_;
  delete[] in_buffer_;
}

// Perform 2-D separable filter.
template <vsip::support_region_type R,
	  edge_handling_type E>
template <typename Block0, typename Block1>
void
Sfilt<R, E>::filter(vsip::const_Matrix<float, Block0> in,
		    vsip::Matrix<float, Block1> out)
VSIP_NOTHROW
{
  using vsip::impl::Any_type;
  using vsip::impl::SYNC_IN;
  using vsip::impl::SYNC_OUT;
  using vsip::index_type;
  using vsip::length_type;
  using vsip::stride_type;
  using vsip::impl::Layout;
  using vsip::impl::Ext_data_dist;
  using vsip::impl::Block_layout;
  using vsip::impl::Cmplx_inter_fmt;
  using vsip::impl::Adjust_layout;

  // PROFILE: Warn if arguments are not entirely on single processor
  // (either as undistributed views or as local views of distr obj).

  length_type const Mr = this->coeff0_.size();
  length_type const Mc = this->coeff1_.size();

  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();

  length_type const Pr = this->output_size_[0].size();
  length_type const Pc = this->output_size_[1].size();

  assert(Pr == out.size(0) && Pc == out.size(1));

  typedef typename Block_layout<Block0>::layout_type LP0;
  typedef typename Block_layout<Block1>::layout_type LP1;

  typedef Layout<2, Any_type, Any_type, Cmplx_inter_fmt> req_LP;

  typedef typename Adjust_layout<float, req_LP, LP0>::type use_LP0;
  typedef typename Adjust_layout<float, req_LP, LP1>::type use_LP1;

  typedef Ext_data_dist<Block0, SYNC_IN,  use_LP0>  in_ext_type;
  typedef Ext_data_dist<Block1, SYNC_OUT, use_LP1> out_ext_type;

  in_ext_type  in_ext (in.block(),  in_buffer_);
  out_ext_type out_ext(out.block(), out_buffer_);

  VSIP_IMPL_PROFILE(pm_in_ext_cost_  += in_ext.cost());
  VSIP_IMPL_PROFILE(pm_out_ext_cost_ += out_ext.cost());

  float *pin    = in_ext.data();
  float *pout   = out_ext.data();

  stride_type in_row_stride    = in_ext.stride(0);
  stride_type in_col_stride    = in_ext.stride(1);
  stride_type out_row_stride   = out_ext.stride(0);
  stride_type out_col_stride   = out_ext.stride(1);

  if (R == vsip::support_full)
  {
    VSIP_IMPL_THROW(std::runtime_error(
      "Separable_filter IPP BE does not implement support_full"));
  }
  else if (R == vsip::support_same)
  {
    VSIP_IMPL_THROW(std::runtime_error(
      "Separable_filter IPP BE does not implement support_same"));
  }
  else if (R == vsip::support_min_zeropad)
  {
    assert(Nr == Pr && Nc == Pc);

    if (in_col_stride == 1 && out_col_stride == 1)
    {
      index_type ro = Mr/2;
      index_type co = Mc/2;

      IppiSize tmp_size = { Pc, Pr - Mr + 1 };
      ippiFilterColumn_32f_C1R(
	pin + ro*in_row_stride,         in_row_stride  * sizeof(float),
	tmp_buffer_ + ro*Pc, Pc * sizeof(float),
	tmp_size,
	pcoeff0_, Mr,
	ro - (Mr % 2 == 0 ? 1 : 0));

      IppiSize out_size = { Pc - Mc + 1, Pr - Mr + 1 };
      ippiFilterRow_32f_C1R(
	tmp_buffer_ + ro*Pc + co, Pc * sizeof(float),
	pout + ro*out_row_stride + co, out_row_stride  * sizeof(float),
	out_size,
	pcoeff1_, Mc,
	co - (Mc % 2 == 0 ? 1 : 0));

      // Zero out boundary
      for (index_type r=0; r<ro; ++r)
	for (index_type c=0; c<Pc; ++c)
	  pout[r*out_row_stride+c] = 0.;
      for (index_type r=Pr-(Mr-ro-1); r<Pr; ++r)
	for (index_type c=0; c<Pc; ++c)
	  pout[r*out_row_stride+c] = 0.;

      for (index_type c=0; c<co; ++c)
	for (index_type r=0; r<Pr; ++r)
	  pout[r*out_row_stride+c] = 0.;
      for (index_type c=Pc-(Mc-co-1); c<Pc; ++c)
	for (index_type r=0; r<Pr; ++r)
	  pout[r*out_row_stride+c] = 0.;
    }
    else
    {
      sfilt_min_zeropad<float>(pcoeff0_, Mr,
			       pcoeff1_, Mc,
			       pin,  in_row_stride, in_col_stride,
			       pout, out_row_stride, out_col_stride,
			       tmp_buffer_,
			       Nr, Nc);
    }
  }
  else // (R == support_min)
  {
    VSIP_IMPL_THROW(std::runtime_error(
      "Separable_filter IPP BE does not implement support_min"));
  }
}

} // namespace vsip_csl::img::impl::ipp
} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img

namespace dispatcher
{
template <vsip::support_region_type R,
	  img::edge_handling_type E,
	  unsigned N,
          vsip::alg_hint_type H>
struct Evaluator<op::sfilt<2, R, E, N, H>, be::intel_ipp, float>
{
  static bool const ct_valid = true;
  typedef img::impl::ipp::Sfilt<R, E> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_SFILT_IPP_HPP
