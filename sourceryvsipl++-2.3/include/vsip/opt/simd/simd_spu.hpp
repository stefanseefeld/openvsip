/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/simd_spu.hpp
    @author  Jules Bergmann
    @date    2007-11-20
    @brief   VSIPL++ Library: SPU SIMD traits.

*/

#ifndef VSIP_OPT_SIMD_SPU_HPP
#define VSIP_OPT_SIMD_SPU_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <spu_mfcio.h>
#include <complex>

#include <vsip/opt/simd/simd_common.hpp>

/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace simd
{

/***********************************************************************
  Cell/B.E. SPU
***********************************************************************/

#define VSIP_IMPL_SPU_LITERAL(_type_, ...) ((_type_){__VA_ARGS__})

#if __BIG_ENDIAN__
#  define VSIP_IMPL_SCALAR_POS(VS) 0
#else
#  define VSIP_IMPL_SCALAR_POS(VS) VS-1
#endif



/***********************************************************************
  Cell/B.E. SPU - signed char vector
***********************************************************************/

template <>
struct Simd_traits<signed char>
{
  typedef signed char            value_type;
  typedef __vector signed char   simd_type;
  typedef __vector unsigned char perm_simd_type;
  typedef __vector unsigned char bool_simd_type;
   
  static int  const vec_size   = 16;
  static bool const is_accel   = true;
  static bool const has_perm   = true;
  static bool const has_div    = false;
  static int  const alignment  = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type,
				0, 0, 0, 0,  0, 0, 0, 0,
				0, 0, 0, 0,  0, 0, 0, 0 );
  }

  static simd_type load(value_type const* addr)
  { return *((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    // Language Extentions for CBEA, section 1.8
    simd_type x0 = *((simd_type*)addr);
    simd_type x1 = *((simd_type*)addr + 1);
    unsigned int shift = (unsigned int)(addr) & 15;
    return spu_or(spu_slqwbyte(x0, shift),
		  spu_rlmaskqwbyte(x1, (signed)(shift - 16)));
  }
  
  static perm_simd_type shift_for_addr(value_type const* addr)
  { 
    typedef __vector unsigned short us_simd_type;
    return ((perm_simd_type)spu_add(
	      (us_simd_type)(spu_splats((unsigned char)(((int)(addr)) & 0xF))),
	      ((us_simd_type){0x0001, 0x0203, 0x0405, 0x0607,
		              0x0809, 0x0A0B, 0x0C0D, 0x0E0F})));
  }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return spu_shuffle(x0, x1, spu_and(sh, 0x1F)); }

  static simd_type load_scalar(value_type value)
  { return (simd_type)si_from_float(value); }

  static simd_type load_scalar_all(value_type value)
  { return spu_splats(value); }

  static void store(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  // SPU cannot add/sub chars
  // static simd_type add(simd_type const& v1, simd_type const& v2)
  // static simd_type sub(simd_type const& v1, simd_type const& v2)

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return spu_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return spu_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return spu_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return spu_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v2, v1); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  {
    bool_simd_type is_lt = spu_cmpgt(v2, v1);
    return spu_nand(is_lt, is_lt);
  }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return ge(v2, v1); }

  static void enter() {}
  static void exit()  {}
};



/***********************************************************************
  Cell/B.E. SPU - vector of signed short
***********************************************************************/

template <>
struct Simd_traits<signed short>
{
  typedef signed short            value_type;
  typedef __vector signed short   simd_type;
  typedef __vector unsigned char  perm_simd_type;
  typedef __vector unsigned short bool_simd_type;
  typedef __vector signed char    pack_simd_type;
  typedef __vector unsigned short count_simd_type;
   
  static int const  vec_size  = 8;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, 0, 0, 0, 0,  0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return *((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    // Language Extentions for CBEA, section 1.8
    simd_type x0 = *((simd_type*)addr);
    simd_type x1 = *((simd_type*)addr + 1);
    unsigned int shift = (unsigned int)(addr) & 15;
    return spu_or(spu_slqwbyte(x0, shift),
		  spu_rlmaskqwbyte(x1, (signed)(shift - 16)));
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  { 
    typedef __vector unsigned short us_simd_type;
    return ((perm_simd_type)spu_add(
	      (us_simd_type)(spu_splats((unsigned char)(((int)(addr)) & 0xF))),
	      ((us_simd_type){0x0001, 0x0203, 0x0405, 0x0607,
		              0x0809, 0x0A0B, 0x0C0D, 0x0E0F})));
  }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return spu_shuffle(x0, x1, spu_and(sh, 0x1F)); }

  static simd_type load_scalar(value_type value)
  { return (simd_type)si_from_short(value); }

  static simd_type load_scalar_all(value_type value)
  { return spu_splats(value); }

  static simd_type load_values(value_type v0, value_type v1,
			       value_type v2, value_type v3,
			       value_type v4, value_type v5,
			       value_type v6, value_type v7)
  { return VSIP_IMPL_SPU_LITERAL(simd_type, v0, v1, v2, v3, v4, v5, v6, v7); }

  static void store(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3,
			  value_type& v4, value_type& v5,
			  value_type& v6, value_type& v7)
  {
    v0 = spu_extract(v, 0);
    v1 = spu_extract(v, 1);
    v2 = spu_extract(v, 2);
    v3 = spu_extract(v, 3);
    v4 = spu_extract(v, 4);
    v5 = spu_extract(v, 5);
    v6 = spu_extract(v, 6);
    v7 = spu_extract(v, 7);
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return spu_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return spu_sub(v1, v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return spu_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return spu_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return spu_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return spu_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v2, v1); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  {
    bool_simd_type is_lt = spu_cmpgt(v2, v1);
    return spu_nand(is_lt, is_lt);
  }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return ge(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { // return vec_pack(v1, v2);
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    1,  3,  5,  7,  9, 11, 13, 15,
			   17, 19, 21, 23, 25, 27, 29, 31);
    return (pack_simd_type)spu_shuffle(v1, v2, shuf);
  }

  static simd_type shiftl(simd_type const& v1, count_simd_type const& v2)
  { return spu_sl(v1, v2); }

  template <int shift>
  static simd_type shiftr(simd_type const& v1)
  { return spu_rlmaska(v1, -shift); }

  static void enter() {}
  static void exit()  {}
};



/***********************************************************************
  Cell/B.E. SPU - vector of unsigned short
***********************************************************************/

template <>
struct Simd_traits<unsigned short>
{
  typedef unsigned short          value_type;
  typedef __vector unsigned short simd_type;
  typedef __vector unsigned char  perm_simd_type;
  typedef __vector unsigned short bool_simd_type;
  typedef __vector unsigned char  pack_simd_type;
   
  static int const  vec_size  = 8;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, 0, 0, 0, 0,  0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return *((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    // Language Extentions for CBEA, section 1.8
    simd_type x0 = *((simd_type*)addr);
    simd_type x1 = *((simd_type*)addr + 1);
    unsigned int shift = (unsigned int)(addr) & 15;
    return spu_or(spu_slqwbyte(x0, shift),
		  spu_rlmaskqwbyte(x1, (signed)(shift - 16)));
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  {
    typedef __vector unsigned short us_simd_type;
    return ((perm_simd_type)spu_add(
	      (us_simd_type)(spu_splats((unsigned char)(((int)(addr)) & 0xF))),
	      ((us_simd_type){0x0001, 0x0203, 0x0405, 0x0607,
		              0x0809, 0x0A0B, 0x0C0D, 0x0E0F})));
  }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return spu_shuffle(x0, x1, spu_and(sh, 0x1F)); }

  static simd_type load_scalar(value_type value)
  { return (simd_type)si_from_ushort(value); }

  static simd_type load_scalar_all(value_type value)
  { return spu_splats(value); }


  static simd_type load_values(value_type v0, value_type v1,
			       value_type v2, value_type v3,
			       value_type v4, value_type v5,
			       value_type v6, value_type v7)
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, v0, v1, v2, v3, v4, v5, v6, v7);
  }

  static void store(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  static value_type extract(simd_type const& v, int pos)
  { return spu_extract(v, pos); }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3,
			  value_type& v4, value_type& v5,
			  value_type& v6, value_type& v7)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec = v;
    v0 = u.val[0];
    v1 = u.val[1];
    v2 = u.val[2];
    v3 = u.val[3];
    v4 = u.val[4];
    v5 = u.val[5];
    v6 = u.val[6];
    v7 = u.val[7];
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return spu_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return spu_sub(v1, v2); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { // return vec_mladd(v1, v2, v3);
    typedef __vector signed short ss_simd_type;
    typedef __vector signed int   si_simd_type;
    return ((simd_type)
	    (spu_shuffle(
	       spu_madd(
		 (ss_simd_type)(spu_rl((vec_uint4)(v1), -16)),
		 (ss_simd_type)(spu_rl((vec_uint4)(v2), -16)),
		 (si_simd_type)(spu_rl((vec_uint4)(v3), -16))),
	       spu_madd((ss_simd_type)v1, (ss_simd_type)v2,
			spu_extend((ss_simd_type)v3)),
	       ((perm_simd_type){ 2,  3, 18, 19,  6,  7, 22, 23,
		                 10, 11, 26, 27, 14, 15, 30, 31}))));
  }


  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return spu_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return spu_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return spu_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return spu_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v2, v1); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  {
    bool_simd_type is_lt = spu_cmpgt(v2, v1);
    return spu_nand(is_lt, is_lt);
  }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return ge(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { // return vec_pack(v1, v2);
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    1,  3,  5,  7,  9, 11, 13, 15,
			   17, 19, 21, 23, 25, 27, 29, 31);
    return (pack_simd_type)spu_shuffle(v1, v2, shuf);
  }

  static void enter() {}
  static void exit()  {}
};



/***********************************************************************
  Cell/B.E. SPU - vector of signed short
***********************************************************************/

template <>
struct Simd_traits<signed int>
{
  typedef signed int             value_type;
  typedef __vector signed int    simd_type;
  typedef __vector unsigned char perm_simd_type;
  typedef __vector unsigned int  bool_simd_type;
  typedef __vector signed short  pack_simd_type;
   
  static int const  vec_size  = 4;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, 0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return *((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    // Language Extentions for CBEA, section 1.8
    simd_type x0 = *((simd_type*)addr);
    simd_type x1 = *((simd_type*)addr + 1);
    unsigned int shift = (unsigned int)(addr) & 15;
    return spu_or(spu_slqwbyte(x0, shift),
		  spu_rlmaskqwbyte(x1, (signed)(shift - 16)));
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  {
    typedef __vector unsigned short us_simd_type;
    return ((perm_simd_type)spu_add(
	      (us_simd_type)(spu_splats((unsigned char)(((int)(addr)) & 0xF))),
	      ((us_simd_type){0x0001, 0x0203, 0x0405, 0x0607,
		              0x0809, 0x0A0B, 0x0C0D, 0x0E0F})));
  }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return spu_shuffle(x0, x1, spu_and(sh, 0x1F)); }

  static simd_type load_scalar(value_type value)
  { return (simd_type)si_from_int(value); }

  static simd_type load_scalar_all(value_type value)
  { return spu_splats(value); }

  static void store(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return spu_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return spu_sub(v1, v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return spu_and(v1, v2); }

  static bool_simd_type band(bool_simd_type const& v1,
			     bool_simd_type const& v2)
  { return spu_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return spu_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return spu_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return spu_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v2, v1); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  {
    bool_simd_type is_lt = spu_cmpgt(v2, v1);
    return spu_nand(is_lt, is_lt);
  }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return ge(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { // return vec_pack(v1, v2);
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    2,  3,  6,  7, 10, 11, 14, 15,
			   18, 19, 22, 23, 26, 27, 30, 31);
    return (pack_simd_type)spu_shuffle(v1, v2, shuf);
  }

  static pack_simd_type pack_shuffle(simd_type const& v1, simd_type const& v2)
  {
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
 			    2,  3, 18, 19,  6,  7, 22, 23,
			   10, 11, 26, 27, 14, 15, 30, 31);
    return (pack_simd_type)spu_shuffle(v1, v2, shuf);
  }

  template <int shift>
  static simd_type shiftr(simd_type const& v1)
  { return spu_rlmask(v1, -shift); }

  static void enter() {}
  static void exit()  {}
};



/***********************************************************************
  Cell/B.E. SPU - vector of unsigned int
***********************************************************************/

template <>
struct Simd_traits<unsigned int>
{
  typedef unsigned int            value_type;
  typedef __vector unsigned int   simd_type;
  typedef __vector unsigned char  perm_simd_type;
  typedef __vector unsigned int   bool_simd_type;
  typedef __vector unsigned short pack_simd_type;
   
  static int const  vec_size  = 4;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, 0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return *((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    // Language Extentions for CBEA, section 1.8
    simd_type x0 = *((simd_type*)addr);
    simd_type x1 = *((simd_type*)addr + 1);
    unsigned int shift = (unsigned int)(addr) & 15;
    return spu_or(spu_slqwbyte(x0, shift),
		  spu_rlmaskqwbyte(x1, (signed)(shift - 16)));
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  {
    typedef __vector unsigned short us_simd_type;
    return ((perm_simd_type)spu_add(
	      (us_simd_type)(spu_splats((unsigned char)(((int)(addr)) & 0xF))),
	      ((us_simd_type){0x0001, 0x0203, 0x0405, 0x0607,
		              0x0809, 0x0A0B, 0x0C0D, 0x0E0F})));
  }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return spu_shuffle(x0, x1, spu_and(sh, 0x1F)); }

  static simd_type load_scalar(value_type value)
  { return (simd_type)si_from_uint(value); }

  static simd_type load_scalar_all(value_type value)
  { return spu_splats(value); }

  static void store(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  static value_type extract(simd_type const& v, int pos)
  { return spu_extract(v, pos); }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3)
  {
    v0 = spu_extract(v, 0);
    v1 = spu_extract(v, 1);
    v2 = spu_extract(v, 2);
    v3 = spu_extract(v, 3);
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return spu_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return spu_sub(v1, v2); }

  // multiply high half-width (or even half-width elements).
  static simd_type mulh(simd_type const& v1, simd_type const& v2)
  { return spu_mule((__vector unsigned short)v1,
		    (__vector unsigned short)v2); }

  // multiply low half-width (or odd half-width elements).
  static simd_type mull(simd_type const& v1, simd_type const& v2)
  { return spu_mulo((__vector unsigned short)v1,
		    (__vector unsigned short)v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return spu_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return spu_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return spu_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return spu_nand(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v2, v1); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  {
    bool_simd_type is_lt = spu_cmpgt(v2, v1);
    return spu_nand(is_lt, is_lt);
  }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return ge(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { // equiv to vec_pack(v1, v2);
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    2,  3,  6,  7, 10, 11, 14, 15,
			   18, 19, 22, 23, 26, 27, 30, 31);
    return (pack_simd_type)spu_shuffle(v1, v2, shuf);
  }

  static __vector float convert_float(simd_type const& v)
  { return spu_convtf(v, 0); }

  static void enter() {}
  static void exit()  {}
};



/***********************************************************************
  Cell/B.E. SPU - vector of float
***********************************************************************/

template <>
struct Simd_traits<float>
{
  typedef float                          value_type;
  typedef __vector float                 simd_type;
  typedef __vector unsigned int          uint_simd_type;
  typedef __vector signed int            sint_simd_type;
  typedef __vector unsigned char         perm_simd_type;
  typedef __vector unsigned int          bool_simd_type;
   
  static int  const vec_size  = 4;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, 0.f, 0.f, 0.f, 0.f);
  }

  static simd_type load(value_type const* addr)
  { return *((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    // Language Extentions for CBEA, section 1.8
    simd_type x0 = *((simd_type*)addr);
    simd_type x1 = *((simd_type*)addr + 1);
    unsigned int shift = (unsigned int)(addr) & 15;
    return spu_or(spu_slqwbyte(x0, shift),
		  spu_rlmaskqwbyte(x1, (signed)(shift - 16)));
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  {
    typedef __vector unsigned short us_simd_type;
    return ((perm_simd_type)spu_add(
	      (us_simd_type)(spu_splats((unsigned char)(((int)(addr)) & 0xF))),
	      ((us_simd_type){0x0001, 0x0203, 0x0405, 0x0607,
		              0x0809, 0x0A0B, 0x0C0D, 0x0E0F})));
  }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return spu_shuffle(x0, x1, spu_and(sh, 0x1F)); }

  static simd_type load_scalar(value_type value)
  { return (simd_type)si_from_float(value); }

  static simd_type load_scalar_all(value_type value)
  { return spu_splats(value); }

  static simd_type load_values(value_type v0, value_type v1,
			       value_type v2, value_type v3)
  {
    return VSIP_IMPL_SPU_LITERAL(simd_type, v0, v1, v2, v3);
  }

  static void store(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  static void store_stream(value_type* addr, simd_type const& vec)
  { *((simd_type*)addr) = vec; }

  static value_type extract(simd_type const& v, int pos)
  { return spu_extract(v, pos); }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3)
  {
    v0 = spu_extract(v, 0);
    v1 = spu_extract(v, 1);
    v2 = spu_extract(v, 2);
    v3 = spu_extract(v, 3);
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return spu_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return spu_sub(v1, v2); }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return spu_madd(v1, v2, zero()); }

  static simd_type div_est(simd_type const& v1, simd_type const& v2)
  { return spu_madd(v1, spu_re(v2), zero()); }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  { return spu_madd(v1, spu_re(v2), zero()); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return spu_madd(v1, v2, v3); }

  static simd_type recip_est(simd_type const& v1)
  { return spu_re(v1); }

  static simd_type recip(simd_type const& v1)
  {
    simd_type one = VSIP_IMPL_SPU_LITERAL(simd_type, 1.f, 1.f, 1.f, 1.f);
    simd_type y0  = spu_re(v1);
    simd_type t   = spu_nmsub(y0, v1, one);
    simd_type y1  = spu_madd(y0, t, y0);
    return y1;
  }

  static simd_type mag(simd_type const& v1)
  { return ((simd_type)(spu_rlmask(spu_sl((uint_simd_type)(v1), 1), -1))); }

  static simd_type min(simd_type const& v1, simd_type const& v2)
  { return spu_sel(v1, v2, spu_cmpgt(v1, v2)); }

  static simd_type max(simd_type const& v1, simd_type const& v2)
  { return spu_sel(v1, v2, spu_cmpgt(v2, v1)); }

  static simd_type band(bool_simd_type const& v1, simd_type const& v2)
  { return spu_and((simd_type)v1, v2); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return spu_cmpgt(v2, v1); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  {
    bool_simd_type is_lt = spu_cmpgt(v2, v1);
    return spu_nand(is_lt, is_lt);
  }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return ge(v2, v1); }

  static simd_type real_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  {
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			   0,   1,  2,  3,  8,  9, 10, 11,
			   16, 17, 18, 19, 24, 25, 26, 27);
    return spu_shuffle(v1, v2, shuf);
  }

  static simd_type imag_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  {
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    4,  5,  6,  7, 12, 13, 14, 15,
			   20, 21, 22, 23, 28, 29, 30, 31);
    return spu_shuffle(v1, v2, shuf);
  }

  static simd_type interleaved_lo_from_split(simd_type const& real,
					     simd_type const& imag)
  { // equiv to vec_mergeh(real, imag);
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    0,  1,  2,  3, 16, 17, 18, 19,
			    4,  5,  6,  7, 20, 21, 22, 23);
    return spu_shuffle(real, imag, shuf);
  }

  static simd_type interleaved_hi_from_split(simd_type const& real,
					     simd_type const& imag)
  { // equiv to spu_mergel(real, imag);
    static __vector unsigned char shuf = 
      VSIP_IMPL_SPU_LITERAL(__vector unsigned char,
			    8,  9, 10, 11, 24, 25, 26, 27,
			   12, 13, 14, 15, 28, 29, 30, 31);
    return spu_shuffle(real, imag, shuf);
  }

  static uint_simd_type convert_uint(simd_type const& v)
  { return spu_convtu(v, 0); }

  template <int shift>
  static sint_simd_type convert_uint(simd_type const& v)
  { return spu_convtu(v, shift); }

  static sint_simd_type convert_sint(simd_type const& v)
  { return spu_convts(v, 0); }

  template <int shift>
  static sint_simd_type convert_sint(simd_type const& v)
  { return spu_convts(v, shift); }


  static void enter() {}
  static void exit()  {}
};

#undef VSIP_IMPL_SPU_LITERAL

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_SIMD_SPU_HPP
