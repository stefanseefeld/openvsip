/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/img/perspective_warp.hpp
    @author  Jules Bergmann
    @date    2007-11-01
    @brief   VSIPL++ Library: Image-processing perspective warp.

*/

#ifndef VSIP_CSL_IMG_PERSPECTIVE_WARP_HPP
#define VSIP_CSL_IMG_PERSPECTIVE_WARP_HPP

#include <vsip/matrix.hpp>
#include <vsip_csl/img/impl/pwarp_common.hpp>
#include <vsip_csl/img/impl/pwarp_gen.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip_csl/img/impl/pwarp_simd.hpp>
#ifdef VSIP_IMPL_CBE_SDK
#  include <vsip_csl/img/impl/pwarp_cbe.hpp>
#endif

namespace vsip_csl
{
namespace dispatcher
{
template <typename C,
	  typename T,
	  img::interpolate_type I,
	  img::transform_dir D,
	  unsigned N,
	  vsip::alg_hint_type A>
struct List<op::pwarp<C, T, I, D, N, A> >
{
  typedef Make_type_list<be::user,
			 be::cbe_sdk,
			 be::mercury_sal,
			 be::simd,
			 be::generic>::type type;
};

} // namespace vsip_csl::dispatcher

namespace img
{

// Perspective warp image processing object.
template <typename            CoeffT,
	  typename            T,
	  interpolate_type    InterpT,
	  transform_dir       T_dir,
	  unsigned            N_times = 0,
	  vsip::alg_hint_type A_hint = vsip::alg_time>
class Perspective_warp
{
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::pwarp<CoeffT,T,InterpT, T_dir, N_times, A_hint> >::type
  backend_type;

public:
  static vsip::dimension_type const dim = 2;

  template <typename Block1>
  Perspective_warp(vsip::const_Matrix<CoeffT, Block1> coeff,
		   vsip::Domain<2> const&             size)
    VSIP_THROW((std::bad_alloc))
    : backend_(coeff, size)
  {}

  template <typename Block1, typename Block2>
  vsip::Matrix<T, Block2>
  operator()(vsip::const_Matrix<T, Block1> in, vsip::Matrix<T, Block2> out)
    VSIP_NOTHROW
  {
    backend_.filter(in, out);
    return out;
  }

  vsip::Domain<dim> const& input_size() const VSIP_NOTHROW 
  { return backend_.input_size();}
  vsip::Domain<dim> const& output_size() const VSIP_NOTHROW
  { return backend_.output_size();}

private:
  Perspective_warp(Perspective_warp const&) VSIP_NOTHROW;
  Perspective_warp& operator=(Perspective_warp const&) VSIP_NOTHROW;

  backend_type backend_;
};

// Perspective warp image processing utility function.

template <typename CoeffT,
	  typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
perspective_warp(
  vsip::const_Matrix<CoeffT, Block1> P,
  vsip::const_Matrix<T, Block2>      in,
  vsip::Matrix<T, Block3>            out)
{
  typedef Perspective_warp<CoeffT, T, interp_linear, forward, 1,
                           vsip::alg_time>
    pwarp_type;

  pwarp_type pwarp(P, vsip::Domain<2>(in.size(0), in.size(1)));
  pwarp(in, out);
}

} // namespace vsip_csl::img

} // namespace vsip_csl

#endif // VSIP_CSL_IMG_PERSPECTIVE_WARP_HPP
