/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/alf_pwarp_ub.cpp
    @author  Jules Bergmann
    @date    2007-11-19
    @brief   VSIPL++ Library: Kernel to compute perspective warp.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <alf_accel.h>
#include <assert.h>
#include <vsip/core/acconfig.hpp>
#include <spu_mfcio.h>
#include <vsip/opt/cbe/pwarp_params.h>

#include <vsip/opt/simd/simd_spu.hpp>

static unsigned char src_img[VSIP_IMPL_CBE_PWARP_BUFFER_SIZE]
	__attribute__ ((aligned (128)));



/***********************************************************************
  Definitions
***********************************************************************/

void initialize(Pwarp_params* pwp)
{
  if (pwp->in_cols == pwp->in_stride_0)
  {
    unsigned int size = pwp->in_rows * pwp->in_cols;
    unsigned long long ea = pwp->ea_in;

    // assert(size <= IMG_SIZE);

    ea += pwp->in_row_0 * pwp->in_stride_0 + pwp->in_col_0;

    unsigned char*     c_in = src_img;
    while (size > 0)
    {
      unsigned int this_size = (size > 16384) ? 16384 : size;
      mfc_get(c_in, ea, this_size, 31, 0, 0);
      c_in += this_size;
      ea   += this_size;
      size -= this_size;
    }

    mfc_write_tag_mask(1<<31);
    mfc_read_tag_status_all();
  }
  else
  {
    unsigned int size = pwp->in_cols;
    unsigned long long ea = pwp->ea_in+
      pwp->in_row_0 * pwp->in_stride_0 + pwp->in_col_0;

    unsigned int r;
    unsigned char* cur_in = src_img;
    for (r=0; r < pwp->in_rows; ++r)
    {
      mfc_get(cur_in, ea, size, 31, 0, 0);
      mfc_write_tag_mask(1<<31);
      mfc_read_tag_status_all();
      ea     += pwp->in_stride_0;
      cur_in += size;
    }
  }
}



void
apply_proj(
  float const* P,
  float        u,
  float        v,
  float*       x,
  float*       y)
{
  float w = (u * P[6] + v * P[7] + P[8]);
  *x      = (u * P[0] + v * P[1] + P[2]) / w;
  *y      = (u * P[3] + v * P[4] + P[5]) / w;
}



void
apply_proj_w(
  float const* P,
  float        u,
  float        v,
  float*       x,
  float*       y,
  float*       w)
{
  *x = (u * P[0] + v * P[1] + P[2]);
  *y = (u * P[3] + v * P[4] + P[5]);
  *w = (u * P[6] + v * P[7] + P[8]);
}



void
pwarp_offset_test_pattern(
  float*               P,
  unsigned char const* in,
  unsigned int         in_r0,
  unsigned int         in_c0,
  unsigned char*       out,
  unsigned int         out_r0,
  unsigned int         out_c0,
  unsigned int         in_rows,
  unsigned int         in_cols,
  unsigned int         out_rows,
  unsigned int         out_cols)
{
  // Test pattern
  unsigned int r, c;
  for (r=0; r<out_rows; ++r)
    for (c=0; c<out_cols; ++c)
    {
      unsigned int rr = r + out_r0;
      unsigned int cc = c + out_c0;
      out[r*out_cols + c] = ((rr & 0x10) ^ (cc & 0x10)) ? 255 : 0;
    }
}


void
pwarp_offset(
  float*               P,
  unsigned char const* in,
  unsigned int         in_r0,
  unsigned int         in_c0,
  unsigned char*       out,
  unsigned int         out_r0,
  unsigned int         out_c0,
  unsigned int         in_rows,
  unsigned int         in_cols,
  unsigned int         out_rows,
  unsigned int         out_cols)
{
  unsigned int r, c;
  for (r=0; r<out_rows; ++r)
    for (c=0; c<out_cols; ++c)
    {
      float x = (float)(c + out_c0);
      float y = (float)(r + out_r0);
      float u, v;
      apply_proj(P, x, y, &u, &v);

      u -= in_c0;
      v -= in_r0;

      if (u >= 0 && u < in_cols-1 &&
	  v >= 0 && v < in_rows-1)
      {
	unsigned int u0 = (unsigned int)(u);
	unsigned int v0 = (unsigned int)(v);

	float u_beta = u - u0;
	float v_beta = v - v0;

	unsigned char x00 = in[(v0+0)*in_cols + u0+0];
	unsigned char x10 = in[(v0+1)*in_cols + u0+0];
	unsigned char x01 = in[(v0+0)*in_cols + u0+1];
	unsigned char x11 = in[(v0+1)*in_cols + u0+1];

	float x0 = (float)((1 - u_beta) * x00 + u_beta * x01);
	float x1 = (float)((1 - u_beta) * x10 + u_beta * x11);

	float x  = (float)((1 - v_beta) * x0  + v_beta * x1);

	out[r*out_cols + c] = (unsigned char)(x);
      }
      else
      {
	out[r*out_cols + c] = 0;
      }
      
    }
}


void
pwarp_offset_simd(
  float*               P,
  unsigned char const* p_in,
  unsigned int         in_r0,
  unsigned int         in_c0,
  unsigned char*       p_out,
  unsigned int         out_r0,
  unsigned int         out_c0,
  unsigned int         in_rows,
  unsigned int         in_cols,
  unsigned int         out_rows,
  unsigned int         out_cols)
{
  typedef unsigned int index_type;
  typedef unsigned int length_type;
  typedef signed int   stride_type;

  typedef float  CoeffT;
  typedef CoeffT AccumT;
  typedef unsigned char T;

  typedef vsip::impl::simd::Simd_traits<CoeffT> simd;
  typedef simd::simd_type              simd_t;
  typedef simd::bool_simd_type         bool_simd_t;

  typedef vsip::impl::simd::Simd_traits<unsigned int>   ui_simd;
  typedef ui_simd::simd_type                            ui_simd_t;
  typedef vsip::impl::simd::Simd_traits<signed int>     si_simd;
  typedef si_simd::simd_type                            si_simd_t;
  typedef vsip::impl::simd::Simd_traits<unsigned short> us_simd;
  typedef us_simd::simd_type                            us_simd_t;
  typedef vsip::impl::simd::Simd_traits<signed short>   ss_simd;
  typedef ss_simd::simd_type                            ss_simd_t;
  typedef vsip::impl::simd::Simd_traits<unsigned char>  uc_simd;
  typedef uc_simd::simd_type                            uc_simd_t;
  typedef vsip::impl::simd::Simd_traits<signed char>    sc_simd;
  typedef sc_simd::simd_type                            sc_simd_t;

  CoeffT      v_clip  = in_rows - 1;
  CoeffT      u_clip  = in_cols - 1;

  CoeffT u_0, v_0, w_0;
  CoeffT u_1, v_1, w_1;
  apply_proj_w(P, 0.,         0., &u_0, &v_0, &w_0);
  apply_proj_w(P, out_cols-1, 0., &u_1, &v_1, &w_1);
  CoeffT u_delta = (u_1 - u_0) / (out_cols-1);
  CoeffT v_delta = (v_1 - v_0) / (out_cols-1);
  CoeffT w_delta = (w_1 - w_0) / (out_cols-1);

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

  stride_type in_stride_0  = in_cols;
  stride_type out_stride_0 = out_cols;

  ui_simd_t vec_in_stride_0 = ui_simd::load_scalar_all(in_stride_0);
  int yet = 0;

  int const fxp_shift = 7;

  ss_simd_t vec_z3_base = ss_simd::load_scalar_all(1 << (fxp_shift-1));
  si_simd_t vec_z3_i_base = si_simd::load_scalar_all(1 << (15+fxp_shift-1));
  us_simd_t vec_fxp_shift = us_simd::load_scalar_all(fxp_shift);
  si_simd_t vec_start = si_simd::load_scalar_all(0x0000);

  for (index_type r=0; r<out_rows; ++r)
  {
    CoeffT y = static_cast<CoeffT>(r);

    CoeffT u_base, v_base, w_base;
    apply_proj_w(P, 0. + out_c0, y + out_r0, &u_base, &v_base, &w_base);

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

    for (index_type c=0; c<out_cols; c+=16)
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

      vec_u0 = simd::sub(vec_u0, spu_splats((float)in_c0));
      vec_u1 = simd::sub(vec_u1, spu_splats((float)in_c0));
      vec_u2 = simd::sub(vec_u2, spu_splats((float)in_c0));
      vec_u3 = simd::sub(vec_u3, spu_splats((float)in_c0));
      vec_v0 = simd::sub(vec_v0, spu_splats((float)in_r0));
      vec_v1 = simd::sub(vec_v1, spu_splats((float)in_r0));
      vec_v2 = simd::sub(vec_v2, spu_splats((float)in_r0));
      vec_v3 = simd::sub(vec_v3, spu_splats((float)in_r0));

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

      bool_simd_t vec_u0_ltc = simd::lt(vec_u0, vec_u_clip);
      bool_simd_t vec_u1_ltc = simd::lt(vec_u1, vec_u_clip);
      bool_simd_t vec_u2_ltc = simd::lt(vec_u2, vec_u_clip);
      bool_simd_t vec_u3_ltc = simd::lt(vec_u3, vec_u_clip);

      bool_simd_t vec_v0_ltc = simd::lt(vec_v0, vec_v_clip);
      bool_simd_t vec_v1_ltc = simd::lt(vec_v1, vec_v_clip);
      bool_simd_t vec_v2_ltc = simd::lt(vec_v2, vec_v_clip);
      bool_simd_t vec_v3_ltc = simd::lt(vec_v3, vec_v_clip);

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

      us_simd_t vec_s01_good = ui_simd::pack(vec_0_good, vec_1_good);
      us_simd_t vec_s23_good = ui_simd::pack(vec_2_good, vec_3_good);
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

      si_simd_t vec_0_k00= simd::convert_sint<15>(
	simd::fma(vec_u0_1_beta, vec_v0_1_beta, vec_05));
      si_simd_t vec_1_k00= simd::convert_sint<15>(
	simd::fma(vec_u1_1_beta, vec_v1_1_beta, vec_05));
      si_simd_t vec_2_k00= simd::convert_sint<15>(
	simd::fma(vec_u2_1_beta, vec_v2_1_beta, vec_05));
      si_simd_t vec_3_k00= simd::convert_sint<15>(
	simd::fma(vec_u3_1_beta, vec_v3_1_beta, vec_05));
      si_simd_t vec_0_k01= simd::convert_sint<15>(
	simd::fma(vec_u0_beta,   vec_v0_1_beta, vec_05));
      si_simd_t vec_1_k01= simd::convert_sint<15>(
	simd::fma(vec_u1_beta,   vec_v1_1_beta, vec_05));
      si_simd_t vec_2_k01= simd::convert_sint<15>(
	simd::fma(vec_u2_beta,   vec_v2_1_beta, vec_05));
      si_simd_t vec_3_k01= simd::convert_sint<15>(
	simd::fma(vec_u3_beta,   vec_v3_1_beta, vec_05));
      si_simd_t vec_0_k10= simd::convert_sint<15>(
	simd::fma(vec_u0_1_beta, vec_v0_beta, vec_05));
      si_simd_t vec_1_k10= simd::convert_sint<15>(
	simd::fma(vec_u1_1_beta, vec_v1_beta, vec_05));
      si_simd_t vec_2_k10= simd::convert_sint<15>(
	simd::fma(vec_u2_1_beta, vec_v2_beta, vec_05));
      si_simd_t vec_3_k10= simd::convert_sint<15>(
	simd::fma(vec_u3_1_beta, vec_v3_beta, vec_05));
      si_simd_t vec_0_k11= simd::convert_sint<15>(
	simd::fma(vec_u0_beta,   vec_v0_beta, vec_05));
      si_simd_t vec_1_k11= simd::convert_sint<15>(
	simd::fma(vec_u1_beta,   vec_v1_beta, vec_05));
      si_simd_t vec_2_k11= simd::convert_sint<15>(
	simd::fma(vec_u2_beta,   vec_v2_beta, vec_05));
      si_simd_t vec_3_k11= simd::convert_sint<15>(
	simd::fma(vec_u3_beta,   vec_v3_beta, vec_05));

      ss_simd_t vec_01_k00 = si_simd::pack(vec_0_k00, vec_1_k00);
      ss_simd_t vec_23_k00 = si_simd::pack(vec_2_k00, vec_3_k00);
      ss_simd_t vec_01_k01 = si_simd::pack(vec_0_k01, vec_1_k01);
      ss_simd_t vec_23_k01 = si_simd::pack(vec_2_k01, vec_3_k01);
      ss_simd_t vec_01_k10 = si_simd::pack(vec_0_k10, vec_1_k10);
      ss_simd_t vec_23_k10 = si_simd::pack(vec_2_k10, vec_3_k10);
      ss_simd_t vec_01_k11 = si_simd::pack(vec_0_k11, vec_1_k11);
      ss_simd_t vec_23_k11 = si_simd::pack(vec_2_k11, vec_3_k11);

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

      T const* p_00 = p_in + off_00;
      T const* p_01 = p_in + off_01;
      T const* p_02 = p_in + off_02;
      T const* p_03 = p_in + off_03;
      T const* p_10 = p_in + off_10;
      T const* p_11 = p_in + off_11;
      T const* p_12 = p_in + off_12;
      T const* p_13 = p_in + off_13;
      T const* p_20 = p_in + off_20;
      T const* p_21 = p_in + off_21;
      T const* p_22 = p_in + off_22;
      T const* p_23 = p_in + off_23;
      T const* p_30 = p_in + off_30;
      T const* p_31 = p_in + off_31;
      T const* p_32 = p_in + off_32;
      T const* p_33 = p_in + off_33;

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

      vec_01_z00 = ss_simd::shiftl(vec_01_z00, vec_fxp_shift);
      vec_23_z00 = ss_simd::shiftl(vec_23_z00, vec_fxp_shift);
      vec_01_z10 = ss_simd::shiftl(vec_01_z10, vec_fxp_shift);
      vec_23_z10 = ss_simd::shiftl(vec_23_z10, vec_fxp_shift);
      vec_01_z01 = ss_simd::shiftl(vec_01_z01, vec_fxp_shift);
      vec_23_z01 = ss_simd::shiftl(vec_23_z01, vec_fxp_shift);
      vec_01_z11 = ss_simd::shiftl(vec_01_z11, vec_fxp_shift);
      vec_23_z11 = ss_simd::shiftl(vec_23_z11, vec_fxp_shift);
 
      // AltiVec
      // ss_simd_t vec_01_z0 = vec_madds(vec_01_k00, vec_01_z00, vec_start);
      // ss_simd_t vec_23_z0 = vec_madds(vec_23_k00, vec_23_z00, vec_start);
      // ss_simd_t vec_01_z1 = vec_madds(vec_01_k01, vec_01_z01, vec_01_z0);
      // ss_simd_t vec_23_z1 = vec_madds(vec_23_k01, vec_23_z01, vec_23_z0);
      // ss_simd_t vec_01_z2 = vec_madds(vec_01_k10, vec_01_z10, vec_01_z1);
      // ss_simd_t vec_23_z2 = vec_madds(vec_23_k10, vec_23_z10, vec_23_z1);
      // ss_simd_t vec_01_z3 = vec_madds(vec_01_k11, vec_01_z11, vec_01_z2);
      // ss_simd_t vec_23_z3 = vec_madds(vec_23_k11, vec_23_z11, vec_23_z2);

      // SPU
      si_simd_t vec_01_z0l = spu_madd(vec_01_k00, vec_01_z00, vec_start);
      si_simd_t vec_23_z0l = spu_madd(vec_23_k00, vec_23_z00, vec_start);
      si_simd_t vec_01_z1l = spu_madd(vec_01_k01, vec_01_z01, vec_01_z0l);
      si_simd_t vec_23_z1l = spu_madd(vec_23_k01, vec_23_z01, vec_23_z0l);
      si_simd_t vec_01_z2l = spu_madd(vec_01_k10, vec_01_z10, vec_01_z1l);
      si_simd_t vec_23_z2l = spu_madd(vec_23_k10, vec_23_z10, vec_23_z1l);
      si_simd_t vec_01_z3l = spu_madd(vec_01_k11, vec_01_z11, vec_01_z2l);
      si_simd_t vec_23_z3l = spu_madd(vec_23_k11, vec_23_z11, vec_23_z2l);

      si_simd_t vec_01_z0h = spu_mhhadd(vec_01_k00, vec_01_z00, vec_start);
      si_simd_t vec_23_z0h = spu_mhhadd(vec_23_k00, vec_23_z00, vec_start);
      si_simd_t vec_01_z1h = spu_mhhadd(vec_01_k01, vec_01_z01, vec_01_z0h);
      si_simd_t vec_23_z1h = spu_mhhadd(vec_23_k01, vec_23_z01, vec_23_z0h);
      si_simd_t vec_01_z2h = spu_mhhadd(vec_01_k10, vec_01_z10, vec_01_z1h);
      si_simd_t vec_23_z2h = spu_mhhadd(vec_23_k10, vec_23_z10, vec_23_z1h);
      si_simd_t vec_01_z3h = spu_mhhadd(vec_01_k11, vec_01_z11, vec_01_z2h);
      si_simd_t vec_23_z3h = spu_mhhadd(vec_23_k11, vec_23_z11, vec_23_z2h);

      vec_01_z3l = si_simd::add(vec_01_z3l, vec_z3_i_base);
      vec_23_z3l = si_simd::add(vec_23_z3l, vec_z3_i_base);
      vec_01_z3h = si_simd::add(vec_01_z3h, vec_z3_i_base);
      vec_23_z3h = si_simd::add(vec_23_z3h, vec_z3_i_base);

      vec_01_z3l = si_simd::shiftr<15+fxp_shift>(vec_01_z3l);
      vec_23_z3l = si_simd::shiftr<15+fxp_shift>(vec_23_z3l);
      vec_01_z3h = si_simd::shiftr<15+fxp_shift>(vec_01_z3h);
      vec_23_z3h = si_simd::shiftr<15+fxp_shift>(vec_23_z3h);

      ss_simd_t vec_01_z3 = si_simd::pack_shuffle(vec_01_z3h, vec_01_z3l);
      ss_simd_t vec_23_z3 = si_simd::pack_shuffle(vec_23_z3h, vec_23_z3l);

      sc_simd_t vec_out = ss_simd::pack(vec_01_z3, vec_23_z3);
      vec_out = sc_simd::band(vec_good, vec_out);

      sc_simd::store((signed char*)p_out, vec_out);
      p_out += 16;
    }
  }
}


extern "C" 
int input(
  void*        context,
  void*        params,
  void*        entries,
  unsigned int current_count,
  unsigned int total_count)
{
  ALF_ACCEL_DTL_BEGIN(entries, ALF_BUF_IN, 0);
  ALF_ACCEL_DTL_END(entries);
  return 0;
}



extern "C" 
int output(
  void*        context,
  void*        params,
  void*        entries,
  unsigned int cur_iter,
  unsigned int tot_iter)
{
  Pwarp_params* pwp = (Pwarp_params*)params;
  alf_data_addr64_t ea;

  // Transfer output.
  ALF_ACCEL_DTL_BEGIN(entries, ALF_BUF_OUT, 0);
  unsigned long length = pwp->out_cols;
  ea = pwp->ea_out + sizeof(unsigned char) *
           ((cur_iter + pwp->out_row_0) * pwp->out_stride_0 + pwp->out_col_0);
  ALF_ACCEL_DTL_ENTRY_ADD(entries, length, ALF_DATA_BYTE, ea);
  ALF_ACCEL_DTL_END(entries);
  return 0;
}

extern "C" 
int kernel(
  void*        context,
  void*        params,
  void*        input,
  void*        output,
  void*        inout,
  unsigned int cur_iter,
  unsigned int tot_iter)
{
  Pwarp_params* pwp = (Pwarp_params *)params;

  if (cur_iter == 0)
    initialize(pwp);

  unsigned char* out = (unsigned char*)output;

  pwarp_offset_simd(
    pwp->P,
    src_img, pwp->in_row_0,             pwp->in_col_0,
    out,     pwp->out_row_0 + cur_iter, pwp->out_col_0,
    pwp->in_rows, pwp->in_cols,
    1, pwp->out_cols);

  return 0;
}

extern "C"
{
ALF_ACCEL_EXPORT_API_LIST_BEGIN
  ALF_ACCEL_EXPORT_API ("input", input);
  ALF_ACCEL_EXPORT_API ("output", output); 
  ALF_ACCEL_EXPORT_API ("kernel", kernel);
ALF_ACCEL_EXPORT_API_LIST_END
}
