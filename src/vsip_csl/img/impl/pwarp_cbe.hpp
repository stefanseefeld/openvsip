/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/img/impl/pwarp_cbe.hpp
    @author  Jules Bergmann
    @date    2007-11-16
    @brief   VSIPL++ Library: Cbe perspective warp transform.
*/

#ifndef VSIP_CSL_IMG_IMPL_PWARP_CBE_HPP
#define VSIP_CSL_IMG_IMPL_PWARP_CBE_HPP

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/opt/cbe/ppu/pwarp.hpp>
#include <vsip_csl/img/impl/pwarp_common.hpp>

namespace vsip_csl
{
namespace img
{
namespace impl
{
namespace cbe
{

template <typename CoeffT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
pwarp_offset(
  vsip::const_Matrix<CoeffT, Block1> P,
  vsip::const_Matrix<T, Block2>      in,
  vsip::index_type                   in_r0,
  vsip::index_type                   in_c0,
  vsip::Matrix<T, Block3>            out,
  vsip::index_type                   out_r0,
  vsip::index_type                   out_c0)
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip_csl::img::impl::apply_proj;

  typedef CoeffT AccumT;

  length_type rows = out.size(0);
  length_type cols = out.size(1);

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      CoeffT x = static_cast<CoeffT>(c + out_c0);
      CoeffT y = static_cast<CoeffT>(r + out_r0);
      CoeffT u, v;
      apply_proj<CoeffT>(P, x, y, u, v);

      u -= in_c0;
      v -= in_r0;

      if (u >= 0 && u < in.size(1)-1 &&
	  v >= 0 && v < in.size(0)-1)
      {
	index_type u0 = static_cast<index_type>(u);
	index_type v0 = static_cast<index_type>(v);

	CoeffT u_beta = u - u0;
	CoeffT v_beta = v - v0;

	T x00 = in(v0,   u0);
	T x10 = in(v0+1, u0+0);
	T x01 = in(v0+0, u0+1);
	T x11 = in(v0+1, u0+1);

	AccumT x0 = (AccumT)((1 - u_beta) * x00 + u_beta * x01);
	AccumT x1 = (AccumT)((1 - u_beta) * x10 + u_beta * x11);

	AccumT x  = (AccumT)((1 - v_beta) * x0  + v_beta * x1);

	// out(r, c) = in((index_type)v, (index_type)u);
	out(r, c) = static_cast<T>(x);
      }
      else
      {
	out(r, c) = 0;
      }
      
    }
}


inline vsip::length_type
quantize_floor(vsip::length_type x, vsip::length_type quantum)
{
  // assert(quantum is power of 2);
  return x & ~(quantum-1);
}

inline vsip::length_type
quantize_ceil(
  vsip::length_type x,
  vsip::length_type quantum,
  vsip::length_type max)
{
  // assert(quantum is power of 2);
  x = (x-1 % quantum == 0) ? x : (x & ~(quantum-1)) + quantum-1;
  if (x > max) x = max;
  return x;
}




template <typename CoeffT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
pwarp_block(
  vsip::const_Matrix<CoeffT, Block1> P,
  vsip::const_Matrix<T, Block2>      in,
  vsip::Matrix<T, Block3>            out)
{
  using vsip::length_type;
  using vsip::index_type;
  using vsip::Domain;
  using vsip_csl::img::impl::apply_proj;
  using std::min;
  using std::max;

  length_type rows = out.size(0);
  length_type cols = out.size(1);

  length_type row_chunk_size = 128;
  length_type col_chunk_size = 128;

  length_type row_quantum = 1;
  length_type col_quantum = 128/sizeof(T);

  for (index_type r=0; r<rows; r += row_chunk_size)
  {
    length_type my_r_size = std::min(row_chunk_size, rows - r);
    for (index_type c=0; c<cols; c += col_chunk_size)
    {
      length_type my_c_size = std::min(col_chunk_size, cols-c);

      CoeffT u00, v00;
      CoeffT u01, v01;
      CoeffT u10, v10;
      CoeffT u11, v11;
      apply_proj<CoeffT>(P, c+0*my_c_size, r+0*my_r_size, u00, v00);
      apply_proj<CoeffT>(P, c+0*my_c_size, r+1*my_r_size, u01, v01);
      apply_proj<CoeffT>(P, c+1*my_c_size, r+0*my_r_size, u10, v10);
      apply_proj<CoeffT>(P, c+1*my_c_size, r+1*my_r_size, u11, v11);

      CoeffT min_u = max(CoeffT(0), min(min(u00, u01), min(u10, u11)));
      CoeffT min_v = max(CoeffT(0), min(min(v00, v01), min(v10, v11)));
      CoeffT max_u = min(CoeffT(in.size(1)-1), max(max(u00, u01),max(u10, u11)));
      CoeffT max_v = min(CoeffT(in.size(0)-1), max(max(v00, v01),max(v10, v11)));

      index_type in_r0 = quantize_floor((index_type)floorf(min_v), row_quantum);
      index_type in_c0 = quantize_floor((index_type)floorf(min_u), col_quantum);
      index_type in_r1 = quantize_ceil((index_type)ceilf(max_v), row_quantum,in.size(0)-1);
      index_type in_c1 = quantize_ceil((index_type)ceilf(max_u), col_quantum,in.size(1)-1);

      Domain<2> in_dom(Domain<1>(in_r0, 1, in_r1 - in_r0 + 1),
		       Domain<1>(in_c0, 1, in_c1 - in_c0 + 1));

      length_type out_r0 = r;
      length_type out_c0 = c;
      Domain<2> out_dom(Domain<1>(out_r0, 1, my_r_size),
			Domain<1>(out_c0, 1, my_c_size));

      pwarp_offset(P,
		   in(in_dom),   in_r0,  in_c0,
		   out(out_dom), out_r0, out_c0);
    }
  }
}

template <typename C, typename T, transform_dir D>
class Pwarp
{
  static vsip::dimension_type const dim = 2;

  // Compile-time constants.
public:
  static interpolate_type const interp_tv    = interp_linear;
  static transform_dir    const transform_tv = D;

  // Constructors, copies, assignments, and destructors.
public:
  template <typename Block1>
  Pwarp(vsip::const_Matrix<C, Block1> coeff,	// coeffs for dimension 0
	vsip::Domain<dim> const&           size)
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
    if (!strcmp(what, "in_dda_cost"))        return pm_in_dda_cost_;
    else if (!strcmp(what, "out_dda_cost"))  return pm_out_dda_cost_;
    else if (!strcmp(what, "non-opt-calls")) return pm_non_opt_calls_;
    else return 0.f;
  }

  template <typename Block0, typename Block1>
  void
  filter(vsip::const_Matrix<T, Block0> in, vsip::Matrix<T, Block1> out)
    VSIP_NOTHROW
  {
    if (out.size(1) <= vsip::impl::cbe::pwarp_block_max_col_size)
      vsip::impl::cbe::pwarp_block(P_, in, out);
    else
      pwarp_block(P_, in, out);
  }

private:
  Pwarp(Pwarp const&) VSIP_NOTHROW;
  Pwarp& operator=(Pwarp const&) VSIP_NOTHROW;

  vsip::Matrix<C> P_;
  vsip::Domain<dim> size_;

  int               pm_non_opt_calls_;
  size_t            pm_in_dda_cost_;
  size_t            pm_out_dda_cost_;
};

} // namespace vsip_csl::img::impl::cbe
} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img

namespace dispatcher
{
template <img::transform_dir D, unsigned N, vsip::alg_hint_type A>
struct Evaluator<op::pwarp<float, unsigned char, img::interp_linear, D, N, A>,
		 be::cbe_sdk>
{
  static bool const ct_valid = true;
  typedef img::impl::cbe::Pwarp<float, unsigned char, D>
  backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_PWARP_CBE_HPP
