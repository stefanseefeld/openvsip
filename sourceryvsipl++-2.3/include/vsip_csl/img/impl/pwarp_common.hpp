/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/img/impl/pwarp_common.hpp
    @author  Jules Bergmann
    @date    2007-11-16
    @brief   VSIPL++ Library: Common perspective warp routines.
*/

#ifndef VSIP_CSL_IMG_IMPL_PWARP_COMMON_HPP
#define VSIP_CSL_IMG_IMPL_PWARP_COMMON_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/signal/types.hpp>

namespace vsip_csl
{
namespace img
{

enum interpolate_type
{
  interp_nearest_neighbor,
  interp_linear,
  interp_cubic,
  interp_super,
  interp_lanczos
};

enum transform_dir
{
  forward,
  inverse
};

namespace impl
{
// Transforms coordinates with projection matrix.
template <typename T,
	  typename CoeffT,
	  typename Block1>
void
apply_proj(
  vsip::const_Matrix<CoeffT, Block1> P,
  T                                  u,
  T                                  v,
  T&                                 x,
  T&                                 y)
{
  T w =  u * P.get(2, 0) + v * P.get(2, 1) + P.get(2,2);
  x   = (u * P.get(0, 0) + v * P.get(0, 1) + P.get(0,2)) / w;
  y   = (u * P.get(1, 0) + v * P.get(1, 1) + P.get(1,2)) / w;
}



// Partially transform coordinates with projection matrix.

template <typename T,
	  typename CoeffT,
	  typename Block1>
void
apply_proj_w(
  vsip::const_Matrix<CoeffT, Block1> P,
  T                                  u,
  T                                  v,
  T&                                 x,
  T&                                 y,
  T&                                 w)
{
  x = u * P.get(0, 0) + v * P.get(0, 1) + P.get(0,2);
  y = u * P.get(1, 0) + v * P.get(1, 1) + P.get(1,2);
  w = u * P.get(2, 0) + v * P.get(2, 1) + P.get(2,2);
}



// Invert projection matrix for purposes of perspective warping.
//
// Needs further scaling by det(P) to be a true inverse.
//
// (Wolberg 1990, Section 3.4.1)

template <typename T,
	  typename Block1,
	  typename Block2>
void
invert_proj(
  vsip::const_Matrix<T, Block1> P,
  vsip::Matrix<T, Block2>       Pi)
{
  Pi(0,0) = P(1,1)*P(2,2) - P(2,1)*P(1,2);
  Pi(0,1) = P(2,1)*P(0,2) - P(0,1)*P(2,2);
  Pi(0,2) = P(0,1)*P(1,2) - P(1,1)*P(0,2);

  Pi(1,0) = P(2,0)*P(1,2) - P(1,0)*P(2,2);
  Pi(1,1) = P(0,0)*P(2,2) - P(2,0)*P(0,2);
  Pi(1,2) = P(1,0)*P(0,2) - P(0,0)*P(1,2);

  Pi(2,0) = P(1,0)*P(2,1) - P(2,0)*P(1,1);
  Pi(2,1) = P(2,0)*P(0,1) - P(0,0)*P(2,1);
  Pi(2,2) = P(0,0)*P(1,1) - P(1,0)*P(0,1);
}



// Inverse projection (Wolberg 1990, Section 3.4.1)

template <typename T,
	  typename Block1>
void
apply_proj_inv(
  vsip::Matrix<T, Block1> P,
  T                       x,
  T                       y,
  T&                      u,
  T&                      v)
{
  vsip::Matrix<T> Pi(3, 3);

  invert_proj(P, Pi);

  apply_proj(Pi, x, y, u, v);
}

} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img
namespace dispatcher
{
namespace op
{
template <typename C,
	  typename T,
	  img::interpolate_type I,
	  img::transform_dir D,
	  unsigned N,
	  vsip::alg_hint_type A>
struct pwarp;
} // namespace vsip_csl::dispatcher::op
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_PWARP_COMMON_HPP
