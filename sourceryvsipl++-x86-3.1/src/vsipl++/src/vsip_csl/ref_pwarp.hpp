/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ref_pwarp.hpp
    @author  Jules Bergmann
    @date    2007-12-12
    @brief   VSIPL++ Library: Image-processing perspective warp,
                              functional reference.

*/

#ifndef VSIP_CSL_REF_PWARP_HPP
#define VSIP_CSL_REF_PWARP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>



/***********************************************************************
  Reference Definitions
***********************************************************************/

namespace vsip_csl
{

namespace ref
{

namespace pwarp_detail
{

template <typename ViewT>
typename ViewT::value_type
get_value(ViewT view, vsip::index_type r, vsip::index_type c)
{
  typedef typename ViewT::value_type T;
  if (r < view.size(0) && c < view.size(1))
    return view.get(r, c);
  else
    return T(-100);
}

} // namespace pwarp_detail

template <typename CoeffT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
pwarp_incremental(
  vsip::const_Matrix<CoeffT, Block1> P,
  vsip::const_Matrix<T, Block2>      in,
  vsip::Matrix<T, Block3>            out)
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip_csl::img::impl::apply_proj_w;
  using vsip_csl::ref::pwarp_detail::get_value;

  typedef CoeffT AccumT;

  CoeffT      v_clip  = in.size(0) - 1;
  CoeffT      u_clip  = in.size(1) - 1;
  length_type rows    = out.size(0);
  length_type cols    = out.size(1);

  CoeffT u_0, v_0, w_0;
  CoeffT u_1, v_1, w_1;
  apply_proj_w<CoeffT>(P, 0.,     0., u_0, v_0, w_0);
  apply_proj_w<CoeffT>(P, cols-1, 0., u_1, v_1, w_1);
  CoeffT u_delta = (u_1 - u_0) / (cols-1);
  CoeffT v_delta = (v_1 - v_0) / (cols-1);
  CoeffT w_delta = (w_1 - w_0) / (cols-1);

  for (index_type r=0; r<rows; ++r)
  {
    CoeffT y = static_cast<CoeffT>(r);

    CoeffT u_base, v_base, w_base;
    apply_proj_w<CoeffT>(P, 0., y, u_base, v_base, w_base);

    for (index_type c=0; c<cols; ++c)
    {
      CoeffT w =  w_base + c*w_delta;
      CoeffT u = (u_base + c*u_delta) / w;
      CoeffT v = (v_base + c*v_delta) / w;

      if (u >= 0 && u <= u_clip && v >= 0 && v <= v_clip)
      {
	index_type u0 = static_cast<index_type>(u);
	index_type v0 = static_cast<index_type>(v);

	CoeffT u_beta = u - u0;
	CoeffT v_beta = v - v0;

	T z00 = get_value(in, v0,   u0);
	T z10 = get_value(in, v0+1, u0+0);
	T z01 = get_value(in, v0+0, u0+1);
	T z11 = get_value(in, v0+1, u0+1);

	AccumT z0 = (AccumT)((1 - u_beta) * z00 + u_beta * z01);
	AccumT z1 = (AccumT)((1 - u_beta) * z10 + u_beta * z11);

	AccumT z  = (AccumT)((1 - v_beta) * z0  + v_beta * z1);

	out.put(r, c, static_cast<T>(z));
      }
      else
      {
	out.put(r, c, 0);
      }
    }
  }
}



template <typename CoeffT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
pwarp(
  vsip::const_Matrix<CoeffT, Block1> P,
  vsip::const_Matrix<T, Block2>      in,
  vsip::Matrix<T, Block3>            out)
{
  using vsip::index_type;
  using vsip::length_type;
  using vsip_csl::img::impl::apply_proj;
  using vsip_csl::ref::pwarp_detail::get_value;

  typedef CoeffT AccumT;

  length_type rows = out.size(0);
  length_type cols = out.size(1);

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      CoeffT x = static_cast<CoeffT>(c);
      CoeffT y = static_cast<CoeffT>(r);
      CoeffT u, v;
      apply_proj<CoeffT>(P, x, y, u, v);

      if (u >= 0 && u <= in.size(1)-1 &&
	  v >= 0 && v <= in.size(0)-1)
      {
	index_type u0 = static_cast<index_type>(u);
	index_type v0 = static_cast<index_type>(v);

	CoeffT u_beta = u - u0;
	CoeffT v_beta = v - v0;

	T x00 = get_value(in, v0,   u0);
	T x10 = get_value(in, v0+1, u0+0);
	T x01 = get_value(in, v0+0, u0+1);
	T x11 = get_value(in, v0+1, u0+1);

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

} // namespace vsip_csl::ref
} // namespace vsip_csl

#endif // VSIP_CSL_REF_PWARP_HPP
