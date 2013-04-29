/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_CSL_IMG_IMPL_SFILT_GEN_HPP
#define VSIP_CSL_IMG_IMPL_SFILT_GEN_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip_csl/img/impl/sfilt_common.hpp>

namespace vsip_csl
{
namespace img
{
namespace impl
{

/// Specialize Sfilt for using ext data.
template <typename T,
	  vsip::support_region_type R,
	  edge_handling_type E>
class Sfilt
{
  static vsip::dimension_type const dim = 2;

public:
  static vsip::support_region_type const support_tv = R;
  static edge_handling_type        const edge_tv    = E;

  template <typename Block1, typename Block2>
  Sfilt(vsip::const_Vector<T, Block1>  coeff0,	// coeffs for dimension 0
	vsip::const_Vector<T, Block2>  coeff1,	// coeffs for dimension 1
	vsip::Domain<dim> const&       input_size)
    VSIP_THROW((std::bad_alloc));

  ~Sfilt() VSIP_NOTHROW;

  vsip::Domain<dim> const &kernel_size() const VSIP_NOTHROW
  { return kernel_size_;}
  vsip::Domain<dim> const &filter_order() const VSIP_NOTHROW
  { return kernel_size_;}
  vsip::Domain<dim> const &input_size() const VSIP_NOTHROW
  { return input_size_;}
  vsip::Domain<dim> const &output_size() const VSIP_NOTHROW
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
  filter(vsip::const_Matrix<T, Block0> in, vsip::Matrix<T, Block1> out) VSIP_NOTHROW;

  typedef vsip::Layout<1, typename vsip::impl::Row_major<dim>::type,
		       vsip::unit_stride,
		       vsip::interleaved_complex>
		layout_type;
  typedef vsip::Vector<T> coeff_view_type;
  typedef vsip::impl::Persistent_data<typename coeff_view_type::block_type,
			       layout_type>
		c_ext_type;

private:
  Sfilt(Sfilt const&) VSIP_NOTHROW;
  Sfilt &operator=(Sfilt const&) VSIP_NOTHROW;

  coeff_view_type   coeff0_;
  coeff_view_type   coeff1_;
  c_ext_type        coeff0_ext_;
  c_ext_type        coeff1_ext_;
  T*                pcoeff0_;
  T*                pcoeff1_;

  vsip::Domain<dim> kernel_size_;
  vsip::Domain<dim> input_size_;
  vsip::Domain<dim> output_size_;

  T*                in_buffer_;
  T*                out_buffer_;
  T*                tmp_buffer_;

  int               pm_non_opt_calls_;
  size_t            pm_in_ext_cost_;
  size_t            pm_out_ext_cost_;
};

/// Construct a convolution object.
template <typename T,
	  vsip::support_region_type R,
	  edge_handling_type E>
template <typename Block1, typename Block2>
Sfilt<T, R, E>::Sfilt(vsip::const_Vector<T, Block1> coeff0, // coeffs for dimension 0
		      vsip::const_Vector<T, Block2> coeff1, // coeffs for dimension 1
		      vsip::Domain<dim> const &input_size)
  VSIP_THROW((std::bad_alloc))
  : coeff0_     (coeff0.size()),
    coeff1_     (coeff1.size()),
    coeff0_ext_ (coeff0_.block(), vsip::dda::in),
    coeff1_ext_ (coeff1_.block(), vsip::dda::in),
    pcoeff0_    (NULL),
    pcoeff1_    (NULL),

    kernel_size_(vsip::Domain<2>(coeff0.size(), coeff1.size())),
    input_size_ (input_size),
    output_size_(sfilt_output_size(R, kernel_size_, input_size)),
    pm_non_opt_calls_ (0)
{
  coeff0_ = coeff0;
  coeff1_ = coeff1;

  coeff0_ext_.begin();
  coeff1_ext_.begin();

  pcoeff0_ = coeff0_ext_.ptr();
  pcoeff1_ = coeff1_ext_.ptr();

  in_buffer_  = new T[input_size_.size()];
  out_buffer_ = new T[output_size_.size()];
  tmp_buffer_ = new T[output_size_.size()];
}

/// Destroy a generic Convolution_impl object.
template <typename T,
	  vsip::support_region_type R,
	  edge_handling_type E>
Sfilt<T, R, E>::~Sfilt() VSIP_NOTHROW
{
  coeff0_ext_.end();
  coeff1_ext_.end();

  delete[] tmp_buffer_;
  delete[] out_buffer_;
  delete[] in_buffer_;
}

// Perform 2-D separable filter.
template <typename T,
	  vsip::support_region_type R,
	  edge_handling_type E>
template <typename Block0, typename Block1>
void
Sfilt<T, R, E>::filter(vsip::const_Matrix<T, Block0> in,
		       vsip::Matrix<T, Block1> out) VSIP_NOTHROW
{
  using vsip::impl::Any_type;
  using vsip::length_type;
  using vsip::stride_type;
  using vsip::Layout;
  using vsip::get_block_layout;
  using vsip::interleaved_complex;
  using vsip::impl::adjust_layout;

  // PROFILE: Warn if arguments are not entirely on single processor
  // (either as undistributed views or as local views of distr obj).

  length_type const Mr = this->coeff0_.size();
  length_type const Mc = this->coeff1_.size();

  length_type const Nr = this->input_size_[0].size();
  length_type const Nc = this->input_size_[1].size();

  assert(this->output_size_[0].size() == out.size(0) && 
	 this->output_size_[1].size() == out.size(1));

  typedef typename get_block_layout<Block0>::type LP0;
  typedef typename get_block_layout<Block1>::type LP1;

  typedef Layout<2, Any_type, any_packing, interleaved_complex> req_LP;

  typedef typename adjust_layout<T, req_LP, LP0>::type use_LP0;
  typedef typename adjust_layout<T, req_LP, LP1>::type use_LP1;

  typedef dda::Data<Block0, dda::in, use_LP0>  in_ext_type;
  typedef dda::Data<Block1, dda::out, use_LP1> out_ext_type;

  in_ext_type  in_ext (in.block(), in_buffer_);
  out_ext_type out_ext(out.block(), out_buffer_);

  VSIP_IMPL_PROFILE(pm_in_ext_cost_  += in_ext.cost());
  VSIP_IMPL_PROFILE(pm_out_ext_cost_ += out_ext.cost());

  T const *pin = in_ext.ptr();
  T *pout = out_ext.ptr();

  stride_type in_row_stride    = in_ext.stride(0);
  stride_type in_col_stride    = in_ext.stride(1);
  stride_type out_row_stride   = out_ext.stride(0);
  stride_type out_col_stride   = out_ext.stride(1);

  if (R == vsip::support_full)
  {
    VSIP_IMPL_THROW(std::runtime_error(
      "Separable_filter generic BE does not implement support_full"));
  }
  else if (R == vsip::support_same)
  {
    VSIP_IMPL_THROW(std::runtime_error(
      "Separable_filter generic BE does not implement support_same"));
  }
  else if (R == vsip::support_min_zeropad)
  {
    assert(Nr == this->output_size_[0].size() &&
	   Nc == this->output_size_[1].size());

    sfilt_min_zeropad<T>(pcoeff0_, Mr,
			 pcoeff1_, Mc,
			 pin,  in_row_stride, in_col_stride,
			 pout, out_row_stride, out_col_stride,
			 tmp_buffer_,
			 Nr, Nc);
  }
  else // (R == support_min)
  {
    VSIP_IMPL_THROW(std::runtime_error(
      "Separable_filter generic BE does not implement support_min"));
  }
}

} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img

namespace dispatcher
{
template <vsip::support_region_type R,
	  img::edge_handling_type E,
	  unsigned N,
          vsip::alg_hint_type H,
	  typename T>
struct Evaluator<op::sfilt<2, R, E, N, H>, be::generic, T>
{
  static bool const ct_valid = true;
  typedef img::impl::Sfilt<T, R, E> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_SFILT_GEN_HPP
