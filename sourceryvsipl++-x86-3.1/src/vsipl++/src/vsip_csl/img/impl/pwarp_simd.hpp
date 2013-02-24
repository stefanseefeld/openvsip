/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/img/impl/pwarp_simd.hpp
    @author  Jules Bergmann
    @date    2007-11-16
    @brief   VSIPL++ Library: SIMD perspective warp transform.
*/

#ifndef VSIP_CSL_IMG_IMPL_PWARP_SIMD_HPP
#define VSIP_CSL_IMG_IMPL_PWARP_SIMD_HPP

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
namespace simd
{
template <typename CoeffT,
	  typename T>
struct Pwarp_impl_simd
{
  static bool const is_avail = false;
};

#ifdef VSIP_IMPL_SIMD_ALTIVEC
template <>
struct Pwarp_impl_simd<float, float>
{
  static bool const is_avail = true;

  typedef float T;
  typedef float CoeffT;

  template <typename Block1,
	    typename Block2,
	    typename Block3>
  static void
  exec(
    vsip::const_Matrix<CoeffT, Block1> P,
    vsip::const_Matrix<T, Block2>      in,
    vsip::Matrix<T, Block3>            out)
  {
  using vsip::index_type;
  using vsip::length_type;
  using vsip::stride_type;

  typedef CoeffT AccumT;

  typedef vsip::impl::simd::Simd_traits<CoeffT> simd;
  typedef typename simd::simd_type              simd_t;
  typedef typename simd::bool_simd_type         bool_simd_t;

  typedef vsip::impl::simd::Simd_traits<unsigned int> ui_simd;
  typedef typename ui_simd::simd_type                 ui_simd_t;

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

  simd_t vec_u_delta = simd::load_scalar_all(4*u_delta);
  simd_t vec_v_delta = simd::load_scalar_all(4*v_delta);
  simd_t vec_w_delta = simd::load_scalar_all(4*w_delta);
  simd_t vec_0       = simd::load_scalar_all(T(0));
  simd_t vec_1       = simd::load_scalar_all(T(1));
  simd_t vec_u_clip  = simd::load_scalar_all(u_clip);
  simd_t vec_v_clip  = simd::load_scalar_all(v_clip);

  simd_t vec_u_delta_ramp = simd::load_values(0*u_delta,
					      1*u_delta,
					      2*u_delta,
					      3*u_delta);
  simd_t vec_v_delta_ramp = simd::load_values(0*v_delta,
					      1*v_delta,
					      2*v_delta,
					      3*v_delta);
  simd_t vec_w_delta_ramp = simd::load_values(0*w_delta,
					      1*w_delta,
					      2*w_delta,
					      3*w_delta);

  vsip::dda::Data<Block2, dda::in> in_data(in.block());
  vsip::dda::Data<Block3, dda::out> out_data(out.block());

  T const *p_in  = in_data.ptr();
  T* p_out = out_data.ptr();
  stride_type in_stride_0            = in_data.stride(0);
  stride_type out_stride_0_remainder = out_data.stride(0) - cols;

  ui_simd_t vec_in_stride_0 = ui_simd::load_scalar_all(in_stride_0);

  for (index_type r=0; r<rows; ++r)
  {
    CoeffT y = static_cast<CoeffT>(r);

    CoeffT u_base, v_base, w_base;
    apply_proj_w<CoeffT>(P, 0., y, u_base, v_base, w_base);

    simd_t vec_u_base;
    simd_t vec_v_base;
    simd_t vec_w_base;

    vec_u_base = simd::add(simd::load_scalar_all(u_base), vec_u_delta_ramp);
    vec_v_base = simd::add(simd::load_scalar_all(v_base), vec_v_delta_ramp);
    vec_w_base = simd::add(simd::load_scalar_all(w_base), vec_w_delta_ramp);

    for (index_type c=0; c<cols; c+=4)
    {
      simd_t vec_w_re = simd::recip(vec_w_base);
      simd_t vec_u = simd::mul(vec_u_base, vec_w_re);
      simd_t vec_v = simd::mul(vec_v_base, vec_w_re);

      bool_simd_t vec_u_ge0 = simd::ge(vec_u, vec_0);
      bool_simd_t vec_v_ge0 = simd::ge(vec_v, vec_0);
      bool_simd_t vec_u_ltc = simd::le(vec_u, vec_u_clip);
      bool_simd_t vec_v_ltc = simd::le(vec_v, vec_v_clip);

      bool_simd_t vec_u_good = ui_simd::band(vec_u_ge0, vec_u_ltc);
      bool_simd_t vec_v_good = ui_simd::band(vec_v_ge0, vec_v_ltc);
      bool_simd_t vec_good   = ui_simd::band(vec_u_good, vec_v_good);

      // Clear u/v if out of bounds.
      vec_u = simd::band(vec_good, vec_u);
      vec_v = simd::band(vec_good, vec_v);

      ui_simd_t vec_u0 = simd::convert_uint(vec_u);
      ui_simd_t vec_v0 = simd::convert_uint(vec_v);

      simd_t vec_u0_f = ui_simd::convert_float(vec_u0);
      simd_t vec_v0_f = ui_simd::convert_float(vec_v0);

      simd_t vec_u_beta = simd::sub(vec_u, vec_u0_f);
      simd_t vec_v_beta = simd::sub(vec_v, vec_v0_f);
      simd_t vec_u_1_beta = simd::sub(vec_1, vec_u_beta);
      simd_t vec_v_1_beta = simd::sub(vec_1, vec_v_beta);

      ui_simd_t vec_offset = ui_simd::add(
	ui_simd::mull(vec_v0, vec_in_stride_0), vec_u0);

      unsigned int off_0, off_1, off_2, off_3;

      ui_simd::extract_all(vec_offset, off_0, off_1, off_2, off_3);
      T const * p_0 = p_in + off_0;
      T const * p_1 = p_in + off_1;
      T const * p_2 = p_in + off_2;
      T const * p_3 = p_in + off_3;

      T z00_0 =  *p_0;
      T z10_0 = *(p_0 + in_stride_0);
      T z01_0 = *(p_0               + 1);
      T z11_0 = *(p_0 + in_stride_0 + 1);
      T z00_1 =  *p_1;
      T z10_1 = *(p_1 + in_stride_0);
      T z01_1 = *(p_1               + 1);
      T z11_1 = *(p_1 + in_stride_0 + 1);
      T z00_2 =  *p_2;
      T z10_2 = *(p_2 + in_stride_0);
      T z01_2 = *(p_2               + 1);
      T z11_2 = *(p_2 + in_stride_0 + 1);
      T z00_3 =  *p_3;
      T z10_3 = *(p_3 + in_stride_0);
      T z01_3 = *(p_3               + 1);
      T z11_3 = *(p_3 + in_stride_0 + 1);

      simd_t vec_z00 = simd::load_values(z00_0, z00_1, z00_2, z00_3);
      simd_t vec_z10 = simd::load_values(z10_0, z10_1, z10_2, z10_3);
      simd_t vec_z01 = simd::load_values(z01_0, z01_1, z01_2, z01_3);
      simd_t vec_z11 = simd::load_values(z11_0, z11_1, z11_2, z11_3);

      simd_t vec_z0 = simd::fma(vec_u_1_beta, vec_z00,
				simd::mul(vec_u_beta,   vec_z01));
      simd_t vec_z1 = simd::fma(vec_u_1_beta, vec_z10,
				simd::mul(vec_u_beta,   vec_z11));

      simd_t vec_z  = simd::fma(vec_v_1_beta, vec_z0,
				simd::mul(vec_v_beta,   vec_z1));

      vec_z = simd::band(vec_good, vec_z);

      simd::store(p_out, vec_z);
      p_out += 4;

      vec_w_base = simd::add(vec_w_base, vec_w_delta);
      vec_u_base = simd::add(vec_u_base, vec_u_delta);
      vec_v_base = simd::add(vec_v_base, vec_v_delta);
    }
    p_out += out_stride_0_remainder;
  }
  }
};
#endif



#ifdef VSIP_IMPL_SIMD_ALTIVEC
template <>
struct Pwarp_impl_simd<float, unsigned char>
{
  static bool const is_avail = true;

  typedef unsigned char T;
  typedef float CoeffT;

  template <typename Block1,
	    typename Block2,
	    typename Block3>
  static void
  exec(
    vsip::const_Matrix<CoeffT, Block1> P,
    vsip::const_Matrix<T, Block2>      in,
    vsip::Matrix<T, Block3>            out)
  {
    using vsip::index_type;
    using vsip::length_type;
    using vsip::stride_type;

    typedef CoeffT AccumT;

    typedef vsip::impl::simd::Simd_traits<CoeffT> simd;
    typedef typename simd::simd_type              simd_t;
    typedef typename simd::bool_simd_type         bool_simd_t;

    typedef vsip::impl::simd::Simd_traits<unsigned int>   ui_simd;
    typedef typename ui_simd::simd_type                   ui_simd_t;
    typedef vsip::impl::simd::Simd_traits<signed int>     si_simd;
    typedef typename si_simd::simd_type                   si_simd_t;
    typedef vsip::impl::simd::Simd_traits<unsigned short> us_simd;
    typedef typename us_simd::simd_type                   us_simd_t;
    typedef vsip::impl::simd::Simd_traits<signed short>   ss_simd;
    typedef typename ss_simd::simd_type                   ss_simd_t;
    typedef vsip::impl::simd::Simd_traits<signed char>    sc_simd;
    typedef typename sc_simd::simd_type                   sc_simd_t;
    
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
    
    simd_t vec_u_delta = simd::load_scalar_all(4*u_delta);
    simd_t vec_v_delta = simd::load_scalar_all(4*v_delta);
    simd_t vec_w_delta = simd::load_scalar_all(4*w_delta);
    simd_t vec_u_delta_16 = simd::load_scalar_all(16*u_delta);
    simd_t vec_v_delta_16 = simd::load_scalar_all(16*v_delta);
    simd_t vec_w_delta_16 = simd::load_scalar_all(16*w_delta);
    simd_t vec_0       = simd::load_scalar_all(T(0));
    simd_t vec_1       = simd::load_scalar_all(T(1));
    simd_t vec_05      = simd::load_scalar_all(0.0);
    simd_t vec_u_clip  = simd::load_scalar_all(u_clip);
    simd_t vec_v_clip  = simd::load_scalar_all(v_clip);
    
    simd_t vec_u_delta_ramp = simd::load_values(0*u_delta,
						1*u_delta,
						2*u_delta,
						3*u_delta);
    simd_t vec_v_delta_ramp = simd::load_values(0*v_delta,
						1*v_delta,
						2*v_delta,
						3*v_delta);
    simd_t vec_w_delta_ramp = simd::load_values(0*w_delta,
						1*w_delta,
						2*w_delta,
						3*w_delta);
    
    vsip::dda::Data<Block2, dda::in> in_data(in.block());
    vsip::dda::Data<Block3, dda::out> out_data(out.block());
    
    T const *p_in  = in_data.ptr();
    T* p_out = out_data.ptr();
    stride_type in_stride_0            = in_data.stride(0);
    stride_type out_stride_0_remainder = out_data.stride(0) - cols;
    
    ui_simd_t vec_in_stride_0 = ui_simd::load_scalar_all(in_stride_0);
    
    ss_simd_t vec_z3_base = ss_simd::load_scalar_all(0x0040);
    us_simd_t vec_shift_7 = us_simd::load_scalar_all(7);
    ss_simd_t vec_start = ss_simd::load_scalar_all(0x0000);
    
    for (index_type r=0; r<rows; ++r)
    {
      CoeffT y = static_cast<CoeffT>(r);
      
      CoeffT u_base, v_base, w_base;
      apply_proj_w<CoeffT>(P, 0., y, u_base, v_base, w_base);
      
      simd_t vec_u0_base = simd::add(simd::load_scalar_all(u_base),
				     vec_u_delta_ramp);
      simd_t vec_v0_base = simd::add(simd::load_scalar_all(v_base),
				     vec_v_delta_ramp);
      simd_t vec_w0_base = simd::add(simd::load_scalar_all(w_base),
				     vec_w_delta_ramp);

      simd_t vec_w1_base = simd::add(vec_w0_base, vec_w_delta);
      simd_t vec_u1_base = simd::add(vec_u0_base, vec_u_delta);
      simd_t vec_v1_base = simd::add(vec_v0_base, vec_v_delta);
      simd_t vec_w2_base = simd::add(vec_w1_base, vec_w_delta);
      simd_t vec_u2_base = simd::add(vec_u1_base, vec_u_delta);
      simd_t vec_v2_base = simd::add(vec_v1_base, vec_v_delta);
      simd_t vec_w3_base = simd::add(vec_w2_base, vec_w_delta);
      simd_t vec_u3_base = simd::add(vec_u2_base, vec_u_delta);
      simd_t vec_v3_base = simd::add(vec_v2_base, vec_v_delta);
      
      for (index_type c=0; c<cols; c+=16)
      {
	simd_t vec_w0_re = simd::recip(vec_w0_base);
	simd_t vec_w1_re = simd::recip(vec_w1_base);
	simd_t vec_w2_re = simd::recip(vec_w2_base);
	simd_t vec_w3_re = simd::recip(vec_w3_base);
	
	simd_t vec_u0    = simd::mul(vec_u0_base, vec_w0_re);
	simd_t vec_v0    = simd::mul(vec_v0_base, vec_w0_re);
	
	simd_t vec_u1    = simd::mul(vec_u1_base, vec_w1_re);
	simd_t vec_v1    = simd::mul(vec_v1_base, vec_w1_re);
	
	simd_t vec_u2    = simd::mul(vec_u2_base, vec_w2_re);
	simd_t vec_v2    = simd::mul(vec_v2_base, vec_w2_re);
	
	simd_t vec_u3    = simd::mul(vec_u3_base, vec_w3_re);
	simd_t vec_v3    = simd::mul(vec_v3_base, vec_w3_re);
	
	vec_w0_base = simd::add(vec_w0_base, vec_w_delta_16);
	vec_w1_base = simd::add(vec_w1_base, vec_w_delta_16);
	vec_w2_base = simd::add(vec_w2_base, vec_w_delta_16);
	vec_w3_base = simd::add(vec_w3_base, vec_w_delta_16);
	vec_u0_base = simd::add(vec_u0_base, vec_u_delta_16);
	vec_u1_base = simd::add(vec_u1_base, vec_u_delta_16);
	vec_u2_base = simd::add(vec_u2_base, vec_u_delta_16);
	vec_u3_base = simd::add(vec_u3_base, vec_u_delta_16);
	vec_v0_base = simd::add(vec_v0_base, vec_v_delta_16);
	vec_v1_base = simd::add(vec_v1_base, vec_v_delta_16);
	vec_v2_base = simd::add(vec_v2_base, vec_v_delta_16);
	vec_v3_base = simd::add(vec_v3_base, vec_v_delta_16);
	
	bool_simd_t vec_u0_ge0 = simd::ge(vec_u0, vec_0);
	bool_simd_t vec_u1_ge0 = simd::ge(vec_u1, vec_0);
	bool_simd_t vec_u2_ge0 = simd::ge(vec_u2, vec_0);
	bool_simd_t vec_u3_ge0 = simd::ge(vec_u3, vec_0);
	
	bool_simd_t vec_v0_ge0 = simd::ge(vec_v0, vec_0);
	bool_simd_t vec_v1_ge0 = simd::ge(vec_v1, vec_0);
	bool_simd_t vec_v2_ge0 = simd::ge(vec_v2, vec_0);
	bool_simd_t vec_v3_ge0 = simd::ge(vec_v3, vec_0);
	
	bool_simd_t vec_u0_ltc = simd::le(vec_u0, vec_u_clip);
	bool_simd_t vec_u1_ltc = simd::le(vec_u1, vec_u_clip);
	bool_simd_t vec_u2_ltc = simd::le(vec_u2, vec_u_clip);
	bool_simd_t vec_u3_ltc = simd::le(vec_u3, vec_u_clip);

	bool_simd_t vec_v0_ltc = simd::le(vec_v0, vec_v_clip);
	bool_simd_t vec_v1_ltc = simd::le(vec_v1, vec_v_clip);
	bool_simd_t vec_v2_ltc = simd::le(vec_v2, vec_v_clip);
	bool_simd_t vec_v3_ltc = simd::le(vec_v3, vec_v_clip);
	
	bool_simd_t vec_u0_good = ui_simd::band(vec_u0_ge0, vec_u0_ltc);
	bool_simd_t vec_u1_good = ui_simd::band(vec_u1_ge0, vec_u1_ltc);
	bool_simd_t vec_u2_good = ui_simd::band(vec_u2_ge0, vec_u2_ltc);
	bool_simd_t vec_u3_good = ui_simd::band(vec_u3_ge0, vec_u3_ltc);
	bool_simd_t vec_v0_good = ui_simd::band(vec_v0_ge0, vec_v0_ltc);
	bool_simd_t vec_v1_good = ui_simd::band(vec_v1_ge0, vec_v1_ltc);
	bool_simd_t vec_v2_good = ui_simd::band(vec_v2_ge0, vec_v2_ltc);
	bool_simd_t vec_v3_good = ui_simd::band(vec_v3_ge0, vec_v3_ltc);
	bool_simd_t vec_0_good  = ui_simd::band(vec_u0_good, vec_v0_good);
	bool_simd_t vec_1_good  = ui_simd::band(vec_u1_good, vec_v1_good);
	bool_simd_t vec_2_good  = ui_simd::band(vec_u2_good, vec_v2_good);
	bool_simd_t vec_3_good  = ui_simd::band(vec_u3_good, vec_v3_good);

	// Clear u/v if out of bounds.
	vec_u0 = simd::band(vec_0_good, vec_u0);
	vec_u1 = simd::band(vec_1_good, vec_u1);
	vec_u2 = simd::band(vec_2_good, vec_u2);
	vec_u3 = simd::band(vec_3_good, vec_u3);
	vec_v0 = simd::band(vec_0_good, vec_v0);
	vec_v1 = simd::band(vec_1_good, vec_v1);
	vec_v2 = simd::band(vec_2_good, vec_v2);
	vec_v3 = simd::band(vec_3_good, vec_v3);
	
// Use this workaround for GNU PPC version >= 4.1
#if __PPC__ && __GNUC__==4 && __GNUC_MINOR__>=1
	us_simd_t vec_s01_good = (us_simd_t)vec_pack(vec_0_good, vec_1_good);
	us_simd_t vec_s23_good = (us_simd_t)vec_pack(vec_2_good, vec_3_good);
#else
	// 071212: ppu-g++ 4.1.1 can't grok this (even though g++ 4.1.1 on
	// a 970FX can).
	us_simd_t vec_s01_good = ui_simd::pack(vec_0_good, vec_1_good);
	us_simd_t vec_s23_good = ui_simd::pack(vec_2_good, vec_3_good);
#endif
	sc_simd_t vec_good     = (sc_simd_t)us_simd::pack(vec_s01_good,
							  vec_s23_good);
	
	ui_simd_t vec_u0_int = simd::convert_uint(vec_u0);
	ui_simd_t vec_u1_int = simd::convert_uint(vec_u1);
	ui_simd_t vec_u2_int = simd::convert_uint(vec_u2);
	ui_simd_t vec_u3_int = simd::convert_uint(vec_u3);
	
	ui_simd_t vec_v0_int = simd::convert_uint(vec_v0);
	ui_simd_t vec_v1_int = simd::convert_uint(vec_v1);
	ui_simd_t vec_v2_int = simd::convert_uint(vec_v2);
	ui_simd_t vec_v3_int = simd::convert_uint(vec_v3);
	
	simd_t vec_u0_f = ui_simd::convert_float(vec_u0_int);
	simd_t vec_u1_f = ui_simd::convert_float(vec_u1_int);
	simd_t vec_u2_f = ui_simd::convert_float(vec_u2_int);
	simd_t vec_u3_f = ui_simd::convert_float(vec_u3_int);
	simd_t vec_v0_f = ui_simd::convert_float(vec_v0_int);
	simd_t vec_v1_f = ui_simd::convert_float(vec_v1_int);
	simd_t vec_v2_f = ui_simd::convert_float(vec_v2_int);
	simd_t vec_v3_f = ui_simd::convert_float(vec_v3_int);
	
	simd_t vec_u0_beta = simd::sub(vec_u0, vec_u0_f);
	simd_t vec_u1_beta = simd::sub(vec_u1, vec_u1_f);
	simd_t vec_u2_beta = simd::sub(vec_u2, vec_u2_f);
	simd_t vec_u3_beta = simd::sub(vec_u3, vec_u3_f);
	simd_t vec_v0_beta = simd::sub(vec_v0, vec_v0_f);
	simd_t vec_v1_beta = simd::sub(vec_v1, vec_v1_f);
	simd_t vec_v2_beta = simd::sub(vec_v2, vec_v2_f);
	simd_t vec_v3_beta = simd::sub(vec_v3, vec_v3_f);
	simd_t vec_u0_1_beta = simd::sub(vec_1, vec_u0_beta);
	simd_t vec_u1_1_beta = simd::sub(vec_1, vec_u1_beta);
	simd_t vec_u2_1_beta = simd::sub(vec_1, vec_u2_beta);
	simd_t vec_u3_1_beta = simd::sub(vec_1, vec_u3_beta);
	simd_t vec_v0_1_beta = simd::sub(vec_1, vec_v0_beta);
	simd_t vec_v1_1_beta = simd::sub(vec_1, vec_v1_beta);
	simd_t vec_v2_1_beta = simd::sub(vec_1, vec_v2_beta);
	simd_t vec_v3_1_beta = simd::sub(vec_1, vec_v3_beta);
	
	si_simd_t vec_0_k00= vec_cts(
	  simd::fma(vec_u0_1_beta, vec_v0_1_beta, vec_05), 15);
	si_simd_t vec_1_k00= vec_cts(
	  simd::fma(vec_u1_1_beta, vec_v1_1_beta, vec_05), 15);
	si_simd_t vec_2_k00= vec_cts(
	  simd::fma(vec_u2_1_beta, vec_v2_1_beta, vec_05), 15);
	si_simd_t vec_3_k00= vec_cts(
	  simd::fma(vec_u3_1_beta, vec_v3_1_beta, vec_05), 15);
	si_simd_t vec_0_k01= vec_cts(
	  simd::fma(vec_u0_beta,   vec_v0_1_beta, vec_05), 15);
	si_simd_t vec_1_k01= vec_cts(
	  simd::fma(vec_u1_beta,   vec_v1_1_beta, vec_05), 15);
	si_simd_t vec_2_k01= vec_cts(
	  simd::fma(vec_u2_beta,   vec_v2_1_beta, vec_05), 15);
	si_simd_t vec_3_k01= vec_cts(
	  simd::fma(vec_u3_beta,   vec_v3_1_beta, vec_05), 15);
	si_simd_t vec_0_k10= vec_cts(
	  simd::fma(vec_u0_1_beta, vec_v0_beta, vec_05), 15);
	si_simd_t vec_1_k10= vec_cts(
	  simd::fma(vec_u1_1_beta, vec_v1_beta, vec_05), 15);
	si_simd_t vec_2_k10= vec_cts(
	  simd::fma(vec_u2_1_beta, vec_v2_beta, vec_05), 15);
	si_simd_t vec_3_k10= vec_cts(
	  simd::fma(vec_u3_1_beta, vec_v3_beta, vec_05), 15);
	si_simd_t vec_0_k11= vec_cts(
	  simd::fma(vec_u0_beta,   vec_v0_beta, vec_05), 15);
	si_simd_t vec_1_k11= vec_cts(
	  simd::fma(vec_u1_beta,   vec_v1_beta, vec_05), 15);
	si_simd_t vec_2_k11= vec_cts(
	  simd::fma(vec_u2_beta,   vec_v2_beta, vec_05), 15);
	si_simd_t vec_3_k11= vec_cts(
	  simd::fma(vec_u3_beta,   vec_v3_beta, vec_05), 15);
	
	ss_simd_t vec_01_k00 = vec_pack(vec_0_k00, vec_1_k00);
	ss_simd_t vec_23_k00 = vec_pack(vec_2_k00, vec_3_k00);
	ss_simd_t vec_01_k01 = vec_pack(vec_0_k01, vec_1_k01);
	ss_simd_t vec_23_k01 = vec_pack(vec_2_k01, vec_3_k01);
	ss_simd_t vec_01_k10 = vec_pack(vec_0_k10, vec_1_k10);
	ss_simd_t vec_23_k10 = vec_pack(vec_2_k10, vec_3_k10);
	ss_simd_t vec_01_k11 = vec_pack(vec_0_k11, vec_1_k11);
	ss_simd_t vec_23_k11 = vec_pack(vec_2_k11, vec_3_k11);
	
	ui_simd_t vec_0_offset = ui_simd::add(
	  ui_simd::mull(vec_v0_int, vec_in_stride_0), vec_u0_int);
	ui_simd_t vec_1_offset = ui_simd::add(
	  ui_simd::mull(vec_v1_int, vec_in_stride_0), vec_u1_int);
	ui_simd_t vec_2_offset = ui_simd::add(
	  ui_simd::mull(vec_v2_int, vec_in_stride_0), vec_u2_int);
	ui_simd_t vec_3_offset = ui_simd::add(
	  ui_simd::mull(vec_v3_int, vec_in_stride_0), vec_u3_int);
	
	unsigned int off_00, off_01, off_02, off_03;
	unsigned int off_10, off_11, off_12, off_13;
	unsigned int off_20, off_21, off_22, off_23;
	unsigned int off_30, off_31, off_32, off_33;

	ui_simd::extract_all(vec_0_offset, off_00, off_01, off_02, off_03);
	ui_simd::extract_all(vec_1_offset, off_10, off_11, off_12, off_13);
	ui_simd::extract_all(vec_2_offset, off_20, off_21, off_22, off_23);
	ui_simd::extract_all(vec_3_offset, off_30, off_31, off_32, off_33);
	
	T const* p_00 = p_in + off_00; assert(off_00 <= rows*cols);
	T const* p_01 = p_in + off_01; assert(off_01 <= rows*cols);
	T const* p_02 = p_in + off_02; assert(off_02 <= rows*cols);
	T const* p_03 = p_in + off_03; assert(off_03 <= rows*cols);
	T const* p_10 = p_in + off_10; assert(off_10 <= rows*cols);
	T const* p_11 = p_in + off_11; assert(off_11 <= rows*cols);
	T const* p_12 = p_in + off_12; assert(off_12 <= rows*cols);
	T const* p_13 = p_in + off_13; assert(off_13 <= rows*cols);
	T const* p_20 = p_in + off_20; assert(off_20 <= rows*cols);
	T const* p_21 = p_in + off_21; assert(off_21 <= rows*cols);
	T const* p_22 = p_in + off_22; assert(off_22 <= rows*cols);
	T const* p_23 = p_in + off_23; assert(off_23 <= rows*cols);
	T const* p_30 = p_in + off_30; assert(off_30 <= rows*cols);
	T const* p_31 = p_in + off_31; assert(off_31 <= rows*cols);
	T const* p_32 = p_in + off_32; assert(off_32 <= rows*cols);
	T const* p_33 = p_in + off_33; assert(off_33 <= rows*cols);

	T z00_00 =  *p_00;
	T z10_00 = *(p_00 + in_stride_0);
	T z01_00 = *(p_00               + 1);
	T z11_00 = *(p_00 + in_stride_0 + 1);
	T z00_01 =  *p_01;
	T z10_01 = *(p_01 + in_stride_0);
	T z01_01 = *(p_01               + 1);
	T z11_01 = *(p_01 + in_stride_0 + 1);
	T z00_02 =  *p_02;
	T z10_02 = *(p_02 + in_stride_0);
	T z01_02 = *(p_02               + 1);
	T z11_02 = *(p_02 + in_stride_0 + 1);
	T z00_03 =  *p_03;
	T z10_03 = *(p_03 + in_stride_0);
	T z01_03 = *(p_03               + 1);
	T z11_03 = *(p_03 + in_stride_0 + 1);
	
	T z00_10 =  *p_10;
	T z10_10 = *(p_10 + in_stride_0);
	T z01_10 = *(p_10               + 1);
	T z11_10 = *(p_10 + in_stride_0 + 1);
	T z00_11 =  *p_11;
	T z10_11 = *(p_11 + in_stride_0);
	T z01_11 = *(p_11               + 1);
	T z11_11 = *(p_11 + in_stride_0 + 1);
	T z00_12 =  *p_12;
	T z10_12 = *(p_12 + in_stride_0);
	T z01_12 = *(p_12               + 1);
	T z11_12 = *(p_12 + in_stride_0 + 1);
	T z00_13 =  *p_13;
	T z10_13 = *(p_13 + in_stride_0);
	T z01_13 = *(p_13               + 1);
	T z11_13 = *(p_13 + in_stride_0 + 1);

	T z00_20 =  *p_20;
	T z10_20 = *(p_20 + in_stride_0);
	T z01_20 = *(p_20               + 1);
	T z11_20 = *(p_20 + in_stride_0 + 1);
	T z00_21 =  *p_21;
	T z10_21 = *(p_21 + in_stride_0);
	T z01_21 = *(p_21               + 1);
	T z11_21 = *(p_21 + in_stride_0 + 1);
	T z00_22 =  *p_22;
	T z10_22 = *(p_22 + in_stride_0);
	T z01_22 = *(p_22               + 1);
	T z11_22 = *(p_22 + in_stride_0 + 1);
	T z00_23 =  *p_23;
	T z10_23 = *(p_23 + in_stride_0);
	T z01_23 = *(p_23               + 1);
	T z11_23 = *(p_23 + in_stride_0 + 1);
	
	T z00_30 =  *p_30;
	T z10_30 = *(p_30 + in_stride_0);
	T z01_30 = *(p_30               + 1);
	T z11_30 = *(p_30 + in_stride_0 + 1);
	T z00_31 =  *p_31;
	T z10_31 = *(p_31 + in_stride_0);
	T z01_31 = *(p_31               + 1);
	T z11_31 = *(p_31 + in_stride_0 + 1);
	T z00_32 =  *p_32;
	T z10_32 = *(p_32 + in_stride_0);
	T z01_32 = *(p_32               + 1);
	T z11_32 = *(p_32 + in_stride_0 + 1);
	T z00_33 =  *p_33;
	T z10_33 = *(p_33 + in_stride_0);
	T z01_33 = *(p_33               + 1);
	T z11_33 = *(p_33 + in_stride_0 + 1);

	ss_simd_t vec_01_z00 = ss_simd::load_values(
	  z00_00, z00_01, z00_02, z00_03,
	  z00_10, z00_11, z00_12, z00_13);
	ss_simd_t vec_23_z00 = ss_simd::load_values(
	  z00_20, z00_21, z00_22, z00_23,
	  z00_30, z00_31, z00_32, z00_33);
	
	ss_simd_t vec_01_z10 = ss_simd::load_values(
	  z10_00, z10_01, z10_02, z10_03,
	  z10_10, z10_11, z10_12, z10_13);
	ss_simd_t vec_23_z10 = ss_simd::load_values(
	  z10_20, z10_21, z10_22, z10_23,
	  z10_30, z10_31, z10_32, z10_33);
	
	ss_simd_t vec_01_z01 = ss_simd::load_values(
	  z01_00, z01_01, z01_02, z01_03,
	  z01_10, z01_11, z01_12, z01_13);
	ss_simd_t vec_23_z01 = ss_simd::load_values(
	  z01_20, z01_21, z01_22, z01_23,
	  z01_30, z01_31, z01_32, z01_33);

	ss_simd_t vec_01_z11 = ss_simd::load_values(
	  z11_00, z11_01, z11_02, z11_03,
	  z11_10, z11_11, z11_12, z11_13);
	ss_simd_t vec_23_z11 = ss_simd::load_values(
	  z11_20, z11_21, z11_22, z11_23,
	  z11_30, z11_31, z11_32, z11_33);
	
	vec_01_z00 = vec_sl(vec_01_z00, vec_shift_7);
	vec_23_z00 = vec_sl(vec_23_z00, vec_shift_7);
	vec_01_z10 = vec_sl(vec_01_z10, vec_shift_7);
	vec_23_z10 = vec_sl(vec_23_z10, vec_shift_7);
	vec_01_z01 = vec_sl(vec_01_z01, vec_shift_7);
	vec_23_z01 = vec_sl(vec_23_z01, vec_shift_7);
	vec_01_z11 = vec_sl(vec_01_z11, vec_shift_7);
	vec_23_z11 = vec_sl(vec_23_z11, vec_shift_7);
	
	ss_simd_t vec_01_z0 = vec_madds(vec_01_k00, vec_01_z00, vec_start);
	ss_simd_t vec_23_z0 = vec_madds(vec_23_k00, vec_23_z00, vec_start);
	ss_simd_t vec_01_z1 = vec_madds(vec_01_k01, vec_01_z01, vec_01_z0);
	ss_simd_t vec_23_z1 = vec_madds(vec_23_k01, vec_23_z01, vec_23_z0);
	ss_simd_t vec_01_z2 = vec_madds(vec_01_k10, vec_01_z10, vec_01_z1);
	ss_simd_t vec_23_z2 = vec_madds(vec_23_k10, vec_23_z10, vec_23_z1);
	ss_simd_t vec_01_z3 = vec_madds(vec_01_k11, vec_01_z11, vec_01_z2);
	ss_simd_t vec_23_z3 = vec_madds(vec_23_k11, vec_23_z11, vec_23_z2);

	// If pixel is on grid (both u,v = {0 or 1}) then weight will
	// -1 do to conversion to signed.  Correct by taking absolute
	// value.
	vec_01_z3 = ss_simd::mag(vec_01_z3);
	vec_23_z3 = ss_simd::mag(vec_23_z3);

	vec_01_z3 = ss_simd::add(vec_01_z3, vec_z3_base);
	vec_23_z3 = ss_simd::add(vec_23_z3, vec_z3_base);
	
	vec_01_z3 = vec_sr(vec_01_z3, vec_shift_7);
	vec_23_z3 = vec_sr(vec_23_z3, vec_shift_7);

	sc_simd_t vec_out = vec_pack(vec_01_z3, vec_23_z3);
	vec_out = vec_and(vec_good, vec_out);
	
	sc_simd::store((signed char*)p_out, vec_out);
	p_out += 16;
      }
      p_out += out_stride_0_remainder;
    }
  }
};
#endif

/// Simd_builtin_tag implementation of Pwarp
template <typename            CoeffT,
	  typename            T,
	  img::transform_dir DirT>
class Pwarp
{
  static vsip::dimension_type const dim = 2;

public:
  static interpolate_type const interp_tv    = interp_linear;
  static transform_dir    const transform_tv = DirT;

  template <typename Block1>
  Pwarp(vsip::const_Matrix<CoeffT, Block1> coeff,	// coeffs for dimension 0
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
    Pwarp_impl_simd<CoeffT, T>::exec(P_, in, out);
  }

private:
  Pwarp(Pwarp const&) VSIP_NOTHROW;
  Pwarp& operator=(Pwarp const&) VSIP_NOTHROW;

  vsip::Matrix<CoeffT>    P_;
  vsip::Domain<dim> size_;
  int               pm_non_opt_calls_;
  size_t            pm_in_dda_cost_;
  size_t            pm_out_dda_cost_;
};

} // namespace vsip_csl::img::impl::simd
} // namespace vsip_csl::img::impl
} // namespace vsip_csl::img

namespace dispatcher
{
template <typename C,
	  typename T,
	  img::transform_dir D,
	  unsigned N,
	  vsip::alg_hint_type A>
struct Evaluator<op::pwarp<C, T, img::interp_linear, D, N, A>,
		 be::simd_builtin>
{
  static bool const ct_valid = img::impl::simd::Pwarp_impl_simd<C, T>::is_avail;
  typedef img::impl::simd::Pwarp<C, T, D> backend_type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_IMPL_PWARP_SIMD_HPP
