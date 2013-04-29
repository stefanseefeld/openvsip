/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_CSL_IMG_IMPL_PWARP_GEN_HPP
#define VSIP_CSL_IMG_IMPL_PWARP_GEN_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip_csl/img/impl/pwarp_common.hpp>

namespace vsip_csl
{
namespace img
{
namespace impl
{

/// Generic implementation of Pwarp_impl.
template <typename C, typename T, transform_dir D>
class Pwarp
{
  static vsip::dimension_type const dim = 2;

public:
  static interpolate_type const interp_tv    = interp_linear;
  static transform_dir    const transform_tv = D;

  template <typename Block1>
  Pwarp(vsip::const_Matrix<C, Block1> coeff,	// coeffs for dimension 0
	vsip::Domain<dim> const &size)
    VSIP_THROW((std::bad_alloc))
    : P_    (3, 3),
    size_ (size),
    pm_non_opt_calls_ (0)
  {
    P_ = coeff;
  }

  vsip::Domain<dim> const& input_size() const VSIP_NOTHROW
  { return size_;}
  vsip::Domain<dim> const& output_size() const VSIP_NOTHROW
  { return size_;}
//  vsip::support_region_type support() const VSIP_NOTHROW
//    { return SuppT; }

  float impl_performance(char const *what) const
  {
    if (!strcmp(what, "in_ext_cost"))        return pm_in_ext_cost_;
    else if (!strcmp(what, "out_ext_cost"))  return pm_out_ext_cost_;
    else if (!strcmp(what, "non-opt-calls")) return pm_non_opt_calls_;
    else return 0.f;
  }

  template <typename Block0, typename Block1>
  void
  filter(vsip::const_Matrix<T, Block0> in, vsip::Matrix<T, Block1> out)
    VSIP_NOTHROW;

private:
  Pwarp(Pwarp const&) VSIP_NOTHROW;
  Pwarp& operator=(Pwarp const&) VSIP_NOTHROW;

  vsip::Matrix<C> P_;
  vsip::Domain<dim> size_;
  int               pm_non_opt_calls_;
  size_t            pm_in_ext_cost_;
  size_t            pm_out_ext_cost_;
};

// Perform 2-D separable filter.
template <typename C, typename T, transform_dir D>
template <typename Block1, typename Block2>
void
Pwarp<C, T, D>::filter(vsip::const_Matrix<T, Block1> in,
		       vsip::Matrix<T,       Block2> out)
VSIP_NOTHROW
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip::stride_type;

  typedef C AccumT;

  C v_clip  = in.size(0) - 1;
  C u_clip  = in.size(1) - 1;
  length_type rows    = out.size(0);
  length_type cols    = out.size(1);

  C u_0, v_0, w_0;
  C u_1, v_1, w_1;
  apply_proj_w<C>(P_, 0.,     0., u_0, v_0, w_0);
  apply_proj_w<C>(P_, cols-1, 0., u_1, v_1, w_1);
  C u_delta = (u_1 - u_0) / (cols-1);
  C v_delta = (v_1 - v_0) / (cols-1);
  C w_delta = (w_1 - w_0) / (cols-1);

  vsip::dda::Data<Block1, dda::in> in_ext(in.block());
  vsip::dda::Data<Block2, dda::out> out_ext(out.block());

  T const *p_in  = in_ext.ptr();
  T* p_out = out_ext.ptr();
  stride_type in_stride_0            = in_ext.stride(0);
  stride_type out_stride_0_remainder = out_ext.stride(0) - cols;

  stride_type max_offset = (rows-1)*in_stride_0 + (cols-1);
  for (index_type r=0; r<rows; ++r)
  {
    C y = static_cast<C>(r);
    
    C u_base, v_base, w_base;
    apply_proj_w<C>(P_, 0., y, u_base, v_base, w_base);
    
    for (index_type c=0; c<cols; ++c)
    {
      C w =  w_base + c*w_delta;
      C u = (u_base + c*u_delta) / w;
      C v = (v_base + c*v_delta) / w;
      
      if (u >= 0 && u <= u_clip && v >= 0 && v <= v_clip)
      {
	index_type u0 = static_cast<index_type>(u);
	index_type v0 = static_cast<index_type>(v);
	
	C u_beta = u - u0;
	C v_beta = v - v0;
	
	T const *p = p_in + v0*in_stride_0 + u0;

	stride_type limit = max_offset - v0*in_stride_0 - u0;

	stride_type off_10 = std::min<stride_type>(in_stride_0,     limit);
	stride_type off_01 = std::min<stride_type>(              1, limit);
	stride_type off_11 = std::min<stride_type>(in_stride_0 + 1, limit);

	T z00 = *p;            // in.get(v0,   u0);
	T z10 = *(p + off_10); // in.get(v0+1, u0+0);
	T z01 = *(p + off_01); // in.get(v0+0, u0+1);
	T z11 = *(p + off_11); // in.get(v0+1, u0+1);
	
	AccumT z0 = (AccumT)((1 - u_beta) * z00 + u_beta * z01);
	AccumT z1 = (AccumT)((1 - u_beta) * z10 + u_beta * z11);
	
	AccumT z  = (AccumT)((1 - v_beta) * z0  + v_beta * z1);
	
	*p_out++ =  static_cast<T>(z);
      }
      else
      {
	*p_out++ = 0;
      }
    }
    p_out += out_stride_0_remainder;
  }
}

} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img

namespace dispatcher
{
template <typename C,
	  typename T,
	  img::transform_dir D,
	  unsigned N,
	  vsip::alg_hint_type A>
struct Evaluator<op::pwarp<C, T, img::interp_linear, D, N, A>, be::generic>
{
  static bool const ct_valid = true;
  typedef img::impl::Pwarp<C, T, D> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_PWARP_GEN_HPP
