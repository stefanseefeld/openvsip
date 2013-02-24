/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/simd.hpp
    @author  Jules Bergmann
    @date    2006-01-25
    @brief   VSIPL++ Library: SIMD traits.

*/

#ifndef VSIP_IMPL_SIMD_HPP
#define VSIP_IMPL_SIMD_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#if __VEC__
#  define VSIP_IMPL_SIMD_ALTIVEC
#  if !_MC_EXEC
#    include <altivec.h>
#    undef vector
#    undef pixel
#    undef bool
#  endif
#else
#  if defined(__SSE__)
#    include <xmmintrin.h>
#    include <emmintrin.h>
#  endif
#endif

#include <complex>
#include <cassert>

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
  AltiVec
***********************************************************************/

// Not all compilers support typedefs with AltiVec vector types:
// As of (date):
//  - Greenhills supports vector typedefs (20060727)
//  - GCC 3.4.4 does not (20060727)
//  - GCC 4.1.1 does not (20061108)

#ifdef VSIP_IMPL_SIMD_ALTIVEC
#  if __ghs__ || __GNUC__ >= 4

#    if __ghs__
#      define VSIP_IMPL_AV_BOOL bool
#      define VSIP_IMPL_AV_LITERAL(_type_, ...) ((_type_)(__VA_ARGS__))
#    else
#      define VSIP_IMPL_AV_BOOL __bool
#      define VSIP_IMPL_AV_LITERAL(_type_, ...) ((_type_){__VA_ARGS__})
#    endif

#if __BIG_ENDIAN__
#  define VSIP_IMPL_SCALAR_POS(VS)  0
#  define VSIP_IMPL_SCALAR_INCR     1
#else
#  define VSIP_IMPL_SCALAR_POS(VS)  VS-1
#  define VSIP_IMPL_SCALAR_INCR     -1
#endif

// PowerPC AltiVec - signed char
template <>
struct Simd_traits<signed char>
{
  typedef signed char                     value_type;
  typedef __vector signed char            simd_type;
  typedef __vector unsigned char          perm_simd_type;
  typedef __vector VSIP_IMPL_AV_BOOL char bool_simd_type;
   
  static unsigned int  const vec_size   = 16;
  static bool const is_accel   = true;
  static bool const has_perm   = true;
  static bool const has_div    = false;
  static unsigned int  const alignment  = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_AV_LITERAL(simd_type,
				0, 0, 0, 0,  0, 0, 0, 0,
				0, 0, 0, 0,  0, 0, 0, 0 );
  }

  static simd_type load(value_type const* addr)
  { return vec_ld(0, (value_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    simd_type              x0 = vec_ld(0,  (value_type*)addr);
    simd_type              x1 = vec_ld(16, (value_type*)addr);
    __vector unsigned char sh = vec_lvsl(0, (value_type*)addr);
    return vec_perm(x0, x1, sh);
  }
  
  static perm_simd_type shift_for_addr(value_type const* addr)
  { return vec_lvsl(0, addr); }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return vec_perm(x0, x1, sh); }

  static simd_type load_scalar(value_type value)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec    = zero();
    u.val[0] = value;
    return u.vec;
  }

  static simd_type load_scalar_all(value_type value)
  { return vec_splat(load_scalar(value), scalar_pos); }

  static void store(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return vec_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return vec_sub(v1, v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return vec_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return vec_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return vec_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v1, v2); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v2, v1); }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v2, v1); }

  static void enter() {}
  static void exit()  {}
};



// PowerPC AltiVec - signed short vector
template <>
struct Simd_traits<signed short>
{
  typedef signed short                value_type;
  typedef __vector signed short       simd_type;
  typedef __vector unsigned char      perm_simd_type;
  typedef __vector VSIP_IMPL_AV_BOOL short bool_simd_type;
  typedef __vector signed char        pack_simd_type;
   
  static unsigned int const  vec_size  = 8;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static unsigned int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_AV_LITERAL(simd_type, 0, 0, 0, 0,  0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return vec_ld(0, (signed short*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    simd_type              x0 = vec_ld(0,  (value_type*)addr);
    simd_type              x1 = vec_ld(16, (value_type*)addr);
    __vector unsigned char sh = vec_lvsl(0, (value_type*)addr);
    return vec_perm(x0, x1, sh);
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  { return vec_lvsl(0, addr); }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return vec_perm(x0, x1, sh); }

  static simd_type load_scalar(value_type value)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec    = zero();
    u.val[0] = value;
    return u.vec;
  }

  static simd_type load_scalar_all(value_type value)
  { return vec_splat(load_scalar(value), scalar_pos); }

  static simd_type load_values(value_type v0, value_type v1,
			       value_type v2, value_type v3,
			       value_type v4, value_type v5,
			       value_type v6, value_type v7)
  {
#if __ghs__
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.val[scalar_pos + 0*VSIP_IMPL_SCALAR_INCR] = v0;
    u.val[scalar_pos + 1*VSIP_IMPL_SCALAR_INCR] = v1;
    u.val[scalar_pos + 2*VSIP_IMPL_SCALAR_INCR] = v2;
    u.val[scalar_pos + 3*VSIP_IMPL_SCALAR_INCR] = v3;
    u.val[scalar_pos + 4*VSIP_IMPL_SCALAR_INCR] = v4;
    u.val[scalar_pos + 5*VSIP_IMPL_SCALAR_INCR] = v5;
    u.val[scalar_pos + 6*VSIP_IMPL_SCALAR_INCR] = v6;
    u.val[scalar_pos + 7*VSIP_IMPL_SCALAR_INCR] = v7;
    return u.vec;
#else
return VSIP_IMPL_AV_LITERAL(simd_type, v0, v1, v2, v3, v4, v5, v6, v7);
#endif
  }

  static void store(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static value_type extract(simd_type const& v, int pos)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    return u.val[pos];
  }

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
  { return vec_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return vec_sub(v1, v2); }

  static simd_type mag(simd_type const& v1)
  { return vec_abs(v1); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return vec_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return vec_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return vec_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v1, v2); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v2, v1); }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { return vec_pack(v1, v2); }

  static void enter() {}
  static void exit()  {}
};



// PowerPC AltiVec - unsigned short vector
template <>
struct Simd_traits<unsigned short>
{
  typedef unsigned short                   value_type;
  typedef __vector unsigned short          simd_type;
  typedef __vector unsigned char           perm_simd_type;
  typedef __vector VSIP_IMPL_AV_BOOL short bool_simd_type;
  typedef __vector unsigned char           pack_simd_type;
   
  static unsigned int const  vec_size  = 8;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static unsigned int const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_AV_LITERAL(simd_type, 0, 0, 0, 0,  0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  // 071116: CBE SDK 2.1 ppu-g++ 4.1.1 thinks vec_ld returns vector
  //         unsigned int.
  { return (simd_type)vec_ld(0, (unsigned short*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    simd_type              x0 = vec_ld(0,  (value_type*)addr);
    simd_type              x1 = vec_ld(16, (value_type*)addr);
    __vector unsigned char sh = vec_lvsl(0, (value_type*)addr);
    return vec_perm(x0, x1, sh);
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  { return vec_lvsl(0, addr); }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return vec_perm(x0, x1, sh); }

  static simd_type load_scalar(value_type value)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec    = zero();
    u.val[0] = value;
    return u.vec;
  }

  static simd_type load_scalar_all(value_type value)
  { return vec_splat(load_scalar(value), scalar_pos); }

  static simd_type load_values(value_type v0, value_type v1,
			       value_type v2, value_type v3,
			       value_type v4, value_type v5,
			       value_type v6, value_type v7)
  {
#if __ghs__
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.val[scalar_pos + 0*VSIP_IMPL_SCALAR_INCR] = v0;
    u.val[scalar_pos + 1*VSIP_IMPL_SCALAR_INCR] = v1;
    u.val[scalar_pos + 2*VSIP_IMPL_SCALAR_INCR] = v2;
    u.val[scalar_pos + 3*VSIP_IMPL_SCALAR_INCR] = v3;
    u.val[scalar_pos + 4*VSIP_IMPL_SCALAR_INCR] = v4;
    u.val[scalar_pos + 5*VSIP_IMPL_SCALAR_INCR] = v5;
    u.val[scalar_pos + 6*VSIP_IMPL_SCALAR_INCR] = v6;
    u.val[scalar_pos + 7*VSIP_IMPL_SCALAR_INCR] = v7;
    return u.vec;
#else
return VSIP_IMPL_AV_LITERAL(simd_type, v0, v1, v2, v3, v4, v5, v6, v7);
#endif
  }

  static void store(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static value_type extract(simd_type const& v, int pos)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    return u.val[pos];
  }

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
  { return vec_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return vec_sub(v1, v2); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return vec_mladd(v1, v2, v3); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static simd_type band(bool_simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return vec_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return vec_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return vec_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v1, v2); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v2, v1); }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { return vec_pack(v1, v2); }

  static void enter() {}
  static void exit()  {}
};



// PowerPC AltiVec - signed short vector
template <>
struct Simd_traits<signed int>
{
  typedef signed int                     value_type;
  typedef __vector signed int            simd_type;
  typedef __vector unsigned char         perm_simd_type;
  typedef __vector VSIP_IMPL_AV_BOOL int bool_simd_type;
  typedef __vector signed short          pack_simd_type;
   
  static unsigned int const vec_size = 4;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static unsigned int const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_AV_LITERAL(simd_type, 0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return vec_ld(0, (value_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    simd_type              x0 = vec_ld(0,  (value_type*)addr);
    simd_type              x1 = vec_ld(16, (value_type*)addr);
    __vector unsigned char sh = vec_lvsl(0, (value_type*)addr);
    return vec_perm(x0, x1, sh);
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  { return vec_lvsl(0, addr); }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return vec_perm(x0, x1, sh); }

  static simd_type load_scalar(value_type value)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec    = zero();
    u.val[0] = value;
    return u.vec;
  }

  static simd_type load_scalar_all(value_type value)
  { return vec_splat(load_scalar(value), scalar_pos); }

  static void store(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static value_type extract(simd_type const& v, int pos)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    return u.val[pos];
  }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    v0 = u.val[0];
    v1 = u.val[1];
    v2 = u.val[2];
    v3 = u.val[3];
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return vec_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return vec_sub(v1, v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static bool_simd_type band(bool_simd_type const& v1,
			     bool_simd_type const& v2)
  { return vec_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return vec_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return vec_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return vec_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v1, v2); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v2, v1); }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { return vec_pack(v1, v2); }

  static void enter() {}
  static void exit()  {}
};



// PowerPC AltiVec - unsigned short vector
template <>
struct Simd_traits<unsigned int>
{
  typedef unsigned int                   value_type;
  typedef __vector unsigned int          simd_type;
  typedef __vector unsigned char         perm_simd_type;
  typedef __vector VSIP_IMPL_AV_BOOL int bool_simd_type;
  typedef __vector unsigned short        pack_simd_type;
   
  static unsigned int const vec_size = 4;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static unsigned int const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_AV_LITERAL(simd_type, 0, 0, 0, 0);
  }

  static simd_type load(value_type const* addr)
  { return vec_ld(0, (value_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    simd_type              x0 = vec_ld(0,  (value_type*)addr);
    simd_type              x1 = vec_ld(16, (value_type*)addr);
    __vector unsigned char sh = vec_lvsl(0, (value_type*)addr);
    return vec_perm(x0, x1, sh);
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  { return vec_lvsl(0, addr); }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return vec_perm(x0, x1, sh); }

  static simd_type load_scalar(value_type value)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec    = zero();
    u.val[0] = value;
    return u.vec;
  }

  static simd_type load_scalar_all(value_type value)
  { return vec_splat(load_scalar(value), scalar_pos); }

  static void store(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static value_type extract(simd_type const& v, int pos)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    return u.val[pos];
  }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    v0 = u.val[0];
    v1 = u.val[1];
    v2 = u.val[2];
    v3 = u.val[3];
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return vec_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return vec_sub(v1, v2); }

  // multiply high half-width (or even half-width elements).
  static simd_type mulh(simd_type const& v1, simd_type const& v2)
  { return vec_mule((__vector unsigned short)v1, (__vector unsigned short)v2); }

  // multiply low half-width (or odd half-width elements).
  static simd_type mull(simd_type const& v1, simd_type const& v2)
  { return vec_mulo((__vector unsigned short)v1, (__vector unsigned short)v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static bool_simd_type band(bool_simd_type const& v1,
			     bool_simd_type const& v2)
  { return vec_and(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return vec_or(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return vec_xor(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return vec_nor(v1, v1); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v1, v2); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v2, v1); }

  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v2, v1); }

  static pack_simd_type pack(simd_type const& v1, simd_type const& v2)
  { return vec_pack(v1, v2); }

  static __vector float convert_float(simd_type const& v)
  { return vec_ctf(v, 0); }

  static void enter() {}
  static void exit()  {}
};



// PowerPC AltiVec - float vector
template <>
struct Simd_traits<float>
{
  typedef float                          value_type;
  typedef __vector float                 simd_type;
  typedef __vector unsigned int          uint_simd_type;
  typedef __vector unsigned char         perm_simd_type;
  typedef __vector VSIP_IMPL_AV_BOOL int bool_simd_type;
   
  static unsigned int  const vec_size  = 4;
  static bool const is_accel  = true;
  static bool const has_perm  = true;
  static bool const has_div   = false;
  static unsigned int  const alignment = 16;

  static unsigned int  const scalar_pos = VSIP_IMPL_SCALAR_POS(vec_size);

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    return VSIP_IMPL_AV_LITERAL(simd_type, 0.f, 0.f, 0.f, 0.f);
  }

  static simd_type load(value_type const* addr)
  { return vec_ld(0, (value_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  {
    simd_type              x0 = vec_ld(0,  (value_type*)addr);
    simd_type              x1 = vec_ld(16, (value_type*)addr);
    __vector unsigned char sh = vec_lvsl(0, (value_type*)addr);
    return vec_perm(x0, x1, sh);
  }

  static perm_simd_type shift_for_addr(value_type const* addr)
  { return vec_lvsl(0, addr); }

  static simd_type perm(simd_type x0, simd_type x1, perm_simd_type sh)
  { return vec_perm(x0, x1, sh); }

  static simd_type load_scalar(value_type value)
  {
#if __ghs__
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = zero();
    u.val[scalar_pos] = value;
    return u.vec;
#else
    return VSIP_IMPL_AV_LITERAL(simd_type, value, 0.f, 0.f, 0.f);
#endif
  }

  static simd_type load_scalar_all(value_type value)
  { return vec_splat(load_scalar(value), scalar_pos); }

  static simd_type load_values(value_type v0, value_type v1,
			       value_type v2, value_type v3)
  {
#if __ghs__
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.val[scalar_pos + 0*VSIP_IMPL_SCALAR_INCR] = v0;
    u.val[scalar_pos + 1*VSIP_IMPL_SCALAR_INCR] = v1;
    u.val[scalar_pos + 2*VSIP_IMPL_SCALAR_INCR] = v2;
    u.val[scalar_pos + 3*VSIP_IMPL_SCALAR_INCR] = v3;
    return u.vec;
#else
    return VSIP_IMPL_AV_LITERAL(simd_type, v0, v1, v2, v3);
#endif
  }

  static void store(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static void store_stream(value_type* addr, simd_type const& vec)
  { vec_st(vec, 0, addr); }

  static value_type extract(simd_type const& v, int pos)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    return u.val[pos];
  }

  static void extract_all(simd_type const& v,
			  value_type& v0, value_type& v1,
			  value_type& v2, value_type& v3)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    v0 = u.val[0];
    v1 = u.val[1];
    v2 = u.val[2];
    v3 = u.val[3];
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return vec_add(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return vec_sub(v1, v2); }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return vec_madd(v1, v2, zero()); }

  static simd_type div_est(simd_type const& v1, simd_type const& v2)
  { return vec_madd(v1, vec_re(v2), zero()); }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  { return vec_madd(v1, vec_re(v2), zero()); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return vec_madd(v1, v2, v3); }

  static simd_type recip_est(simd_type const& v1)
  { return vec_re(v1); }

  static simd_type recip(simd_type const& v1)
  {
    simd_type one = VSIP_IMPL_AV_LITERAL(simd_type, 1.f, 1.f, 1.f, 1.f);
    simd_type y0  = vec_re(v1);
    simd_type t   = vec_nmsub(y0, v1, one);
    simd_type y1  = vec_madd(y0, t, y0);
    return y1;
  }

  static simd_type mag(simd_type const& v1)
  { return vec_abs(v1); }

  static simd_type min(simd_type const& v1, simd_type const& v2)
  { return vec_min(v1, v2); }

  static simd_type max(simd_type const& v1, simd_type const& v2)
  { return vec_max(v1, v2); }

  static simd_type band(bool_simd_type const& v1, simd_type const& v2)
  { return vec_and(v1, v2); }

  static bool_simd_type gt(simd_type const& v1, simd_type const& v2)
  { return vec_cmpgt(v1, v2); }

  static bool_simd_type lt(simd_type const& v1, simd_type const& v2)
  { return vec_cmplt(v1, v2); }

  static bool_simd_type ge(simd_type const& v1, simd_type const& v2)
  { return vec_cmpge(v1, v2); }

  // 070505: ppu-g++ 4.1.1 confused by return type for vec_cmple
  //         (but regular g++ 4.1.1 OK).  Use vec_cmpgt instead.
  static bool_simd_type le(simd_type const& v1, simd_type const& v2)
  { return vec_cmpge(v2, v1); }

  static simd_type real_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  {
    static __vector unsigned char shuf = 
      VSIP_IMPL_AV_LITERAL(__vector unsigned char,
			   0,   1,  2,  3,  8,  9, 10, 11,
			   16, 17, 18, 19, 24, 25, 26, 27);
    return vec_perm(v1, v2, shuf);
  }

  static simd_type imag_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  {
    static __vector unsigned char shuf = 
      VSIP_IMPL_AV_LITERAL(__vector unsigned char,
			    4,  5,  6,  7, 12, 13, 14, 15,
			   20, 21, 22, 23, 28, 29, 30, 31);
    return vec_perm(v1, v2, shuf);
  }

  static simd_type interleaved_lo_from_split(simd_type const& real,
					     simd_type const& imag)
  { return vec_mergeh(real, imag); }

  static simd_type interleaved_hi_from_split(simd_type const& real,
					     simd_type const& imag)
  { return vec_mergel(real, imag); }

  static uint_simd_type convert_uint(simd_type const& v)
  { return vec_ctu(v, 0); }

  static void enter() {}
  static void exit()  {}
};
#    undef VSIP_IMPL_AV_BOOL
#    undef VSIP_IMPL_AV_LITERAL
#  endif
#endif



/***********************************************************************
  SSE
***********************************************************************/

#ifdef __SSE__
template <>
struct Simd_traits<signed char> 
{
  typedef signed char	value_type;
  typedef __m128i	simd_type;
   
  static unsigned int const vec_size = 16;
  static bool const is_accel   = true;
  static bool const has_perm   = false;
  static bool const has_div    = false;
  static unsigned int const alignment  = 16;
  static unsigned int const scalar_pos = 0;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  { return _mm_setzero_si128(); }

  static simd_type load(value_type* addr)
  { return _mm_load_si128((simd_type*)addr); }

  static simd_type load_unaligned(value_type* addr)
  { return _mm_loadu_si128((simd_type*)addr); }

  static simd_type load_scalar(value_type value)
  { return _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, value); }

  static simd_type load_scalar_all(value_type value)
  { return _mm_set1_epi8(value); }

  static void store(value_type* addr, simd_type const& vec)
  { _mm_store_si128((simd_type*)addr, vec); }

  static void store_stream(value_type* addr, simd_type const& vec)
  { _mm_store_si128((simd_type*)addr, vec); }
  // { __builtin_ia32_movntps((simd_type*)addr, vec); }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return _mm_add_epi8(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return _mm_sub_epi8(v1, v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return _mm_and_si128(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return _mm_or_si128(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return _mm_xor_si128(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return bxor(v1, load_scalar_all(0xFF)); }

  static void enter() {}
  static void exit()  {}
};



template <>
struct Simd_traits<short> 
{
  typedef short		value_type;
  typedef __m128i	simd_type;
   
  static unsigned int const vec_size = 8;
  static bool const is_accel   = true;
  static bool const has_perm   = false;
  static bool const has_div    = false;
  static unsigned int const alignment  = 16;
  static unsigned int const scalar_pos = 0;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  { return _mm_setzero_si128(); }

  static simd_type load(value_type const* addr)
  { return _mm_load_si128((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  { return _mm_loadu_si128((simd_type*)addr); }

  static simd_type load_scalar(value_type value)
  { return _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, value); }

  static simd_type load_scalar_all(value_type value)
  { return _mm_set1_epi16(value); }

  static void store(value_type* addr, simd_type const& vec)
  { _mm_store_si128((simd_type*)addr, vec); }

  static void store_stream(value_type* addr, simd_type const& vec)
  { _mm_store_si128((simd_type*)addr, vec); }
  // { __builtin_ia32_movntps((simd_type*)addr, vec); }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return _mm_add_epi16(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return _mm_sub_epi16(v1, v2); }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return _mm_mullo_epi16(v1, v2); }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  {
    // PROFILE - EXPENSIVE
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u1, u2, r;
    u1.vec = v1;
    u2.vec = v2;
    for (unsigned int i = 0; i < vec_size; ++i)
      r.val[i] = u1.val[i]/u2.val[i];
    return r.vec;
  }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return _mm_and_si128(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return _mm_or_si128(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return _mm_xor_si128(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return bxor(v1, load_scalar_all(0xFFFF)); }

#if 0
  static simd_type extend(simd_type const& v)
  { return _mm_shuffle_ps(v, v, 0x00); }

  static simd_type real_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_ps(v1, v2, 0x88); }

  static simd_type imag_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_ps(v1, v2, 0xDD); }
#endif

  static simd_type interleaved_lo_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpacklo_epi16(real, imag); }

  static simd_type interleaved_hi_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpackhi_epi16(real, imag); }

  static simd_type pack(simd_type const& v1, simd_type const& v2)
  { return _mm_packs_epi16(v1, v2); }

  static void enter() {}
  static void exit()  {}
};



template <>
struct Simd_traits<int>
{
  typedef int		value_type;
  typedef __m128i	simd_type;
   
  static unsigned int const vec_size = 4;
  static bool const is_accel   = true;
  static bool const has_perm   = false;
  static bool const has_div    = false;
  static unsigned int const alignment  = 16;
  static unsigned int const scalar_pos = 0;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  { return _mm_setzero_si128(); }

  static simd_type load(value_type const* addr)
  { return _mm_load_si128((simd_type*)addr); }

  static simd_type load_unaligned(value_type const* addr)
  { return _mm_loadu_si128((simd_type*)addr); }

  static simd_type load_scalar(value_type value)
  { return _mm_set_epi32(0, 0, 0, value); }

  static simd_type load_scalar_all(value_type value)
  { return _mm_set1_epi32(value); }

  static void store(value_type* addr, simd_type const& vec)
  { _mm_store_si128((simd_type*)addr, vec); }

  static void store_stream(value_type* addr, simd_type const& vec)
  { _mm_store_si128((simd_type*)addr, vec); }
  // { __builtin_ia32_movntps((simd_type*)addr, vec); }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return _mm_add_epi32(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return _mm_sub_epi32(v1, v2); }

  static simd_type band(simd_type const& v1, simd_type const& v2)
  { return _mm_and_si128(v1, v2); }

  static simd_type bor(simd_type const& v1, simd_type const& v2)
  { return _mm_or_si128(v1, v2); }

  static simd_type bxor(simd_type const& v1, simd_type const& v2)
  { return _mm_xor_si128(v1, v2); }

  static simd_type bnot(simd_type const& v1)
  { return bxor(v1, load_scalar_all(0xFFFFFFFF)); }

#if 0
  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return _mm_mul_epi32(v1, v2); }

  static simd_type extend(simd_type const& v)
  { return _mm_shuffle_ps(v, v, 0x00); }

  static simd_type real_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_ps(v1, v2, 0x88); }

  static simd_type imag_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_ps(v1, v2, 0xDD); }
#endif

  static simd_type interleaved_lo_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpacklo_epi32(real, imag); }

  static simd_type interleaved_hi_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpackhi_epi32(real, imag); }

  static simd_type pack(simd_type const& v1, simd_type const& v2)
  { return _mm_packs_epi32(v1, v2); }

  static void enter() {}
  static void exit()  {}
};



// SSE - float vector
template <>
struct Simd_traits<float>
{
  typedef float		value_type;
  typedef __m128	simd_type;
   
  static unsigned int const vec_size = 4;
  static bool const is_accel   = true;
  static bool const has_perm   = false;
  static bool const has_div    = true;
  static unsigned int const alignment  = 16;
  static unsigned int const scalar_pos = 0;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  { return _mm_setzero_ps(); }

  static simd_type load(value_type const* addr)
  { return _mm_load_ps(addr); }

  static simd_type load_unaligned(value_type const* addr)
  { return _mm_loadu_ps(addr); }

  static simd_type load_scalar(value_type value)
  { return _mm_load_ss(&value); }

  static simd_type load_scalar_all(value_type value)
  { return _mm_load1_ps(&value); }

  static void store(value_type* addr, simd_type const& vec)
  { _mm_store_ps(addr, vec); }

  static void store_stream(value_type* addr, simd_type const& vec)
  {
#if __amd64__
    // amd64 does not support streaming (emprically on csldemo1)
    _mm_store_ps(addr, vec);
#else
  _mm_stream_ps(addr, vec);
#endif
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return _mm_add_ps(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return _mm_sub_ps(v1, v2); }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return _mm_mul_ps(v1, v2); }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  { return _mm_div_ps(v1, v2); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return add(mul(v1,v2),v3); }

  static simd_type mag(simd_type const& v1)
  {
    simd_type mask = (simd_type)Simd_traits<int>::load_scalar_all(0x7fffffff);
    return _mm_and_ps(mask, v1);
  }

  static simd_type min(simd_type const& v1, simd_type const& v2)
  { return _mm_min_ps(v1, v2); }

  static simd_type max(simd_type const& v1, simd_type const& v2)
  { return _mm_max_ps(v1, v2); }

  static simd_type gt(simd_type const& v1, simd_type const& v2)
  { return _mm_cmpgt_ps(v1, v2); }

  static simd_type lt(simd_type const& v1, simd_type const& v2)
  { return _mm_cmplt_ps(v1, v2); }

  static simd_type ge(simd_type const& v1, simd_type const& v2)
  { return _mm_cmpge_ps(v1, v2); }

  static simd_type le(simd_type const& v1, simd_type const& v2)
  { return _mm_cmple_ps(v1, v2); }

  static int sign_mask(simd_type const& v1)
  { return _mm_movemask_ps(v1); }

  static simd_type extend(simd_type const& v)
  { return _mm_shuffle_ps(v, v, 0x00); }

  static simd_type real_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_ps(v1, v2, 0x88); }

  static simd_type imag_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_ps(v1, v2, 0xDD); }

  static simd_type interleaved_lo_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpacklo_ps(real, imag); }

  static simd_type interleaved_hi_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpackhi_ps(real, imag); }

  static void enter() {}
  static void exit()  {}
};


#if defined(__SSE2__)
// SSE2 - double vector
template <>
struct Simd_traits<double>
{
  typedef double	value_type;
  typedef __m128d	simd_type;
   
  static unsigned int const vec_size = 2;
  static bool const is_accel   = true;
  static bool const has_perm   = false;
  static bool const has_div    = true;
  static unsigned int const alignment  = 16;
  static unsigned int const scalar_pos = 0;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  { return _mm_setzero_pd(); }

  static simd_type load(value_type const* addr)
  { return _mm_load_pd(addr); }

  static simd_type load_unaligned(value_type const* addr)
  { return _mm_loadu_pd(addr); }

  static simd_type load_scalar(value_type value)
  { return _mm_load_sd(&value); }

  static simd_type load_scalar_all(value_type value)
  { return _mm_load1_pd(&value); }

  static void store(value_type* addr, simd_type const& vec)
  { _mm_store_pd(addr, vec); }

  static void store_stream(value_type* addr, simd_type const& vec)
  {
#if __amd64__
    // amd64 does not support streaming (emprically on csldemo1)
    _mm_store_pd(addr, vec);
#else
  _mm_stream_pd(addr, vec);
#endif
  }

  static value_type extract(simd_type const& v, int pos)
  {
    union
    {
      simd_type  vec;
      value_type val[vec_size];
    } u;
    u.vec             = v;
    return u.val[pos];
  }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return _mm_add_pd(v1, v2); }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return _mm_sub_pd(v1, v2); }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return _mm_mul_pd(v1, v2); }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  { return _mm_div_pd(v1, v2); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return add(mul(v1,v2),v3); }

  static simd_type mag(simd_type const& v1)
  {
    simd_type mask = (simd_type)_mm_set_epi32(0x7fffffff, 0xffffffff,
					      0x7fffffff, 0xffffffff);
    return _mm_and_pd(mask, v1);
  }

  static simd_type min(simd_type const& v1, simd_type const& v2)
  { return _mm_min_pd(v1, v2); }

  static simd_type max(simd_type const& v1, simd_type const& v2)
  { return _mm_max_pd(v1, v2); }

  static simd_type gt(simd_type const& v1, simd_type const& v2)
  { return _mm_cmpgt_pd(v1, v2); }

  static simd_type lt(simd_type const& v1, simd_type const& v2)
  { return _mm_cmplt_pd(v1, v2); }

  static simd_type ge(simd_type const& v1, simd_type const& v2)
  { return _mm_cmpge_pd(v1, v2); }

  static simd_type le(simd_type const& v1, simd_type const& v2)
  { return _mm_cmple_pd(v1, v2); }

  static int sign_mask(simd_type const& v1)
  { return _mm_movemask_pd(v1); }

  static simd_type extend(simd_type const& v)
  { return _mm_shuffle_pd(v, v, 0x0); }

  static simd_type real_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_pd(v1, v2, 0x0); }

  static simd_type imag_from_interleaved(simd_type const& v1,
					 simd_type const& v2)
  { return _mm_shuffle_pd(v1, v2, 0x3); }

  static simd_type interleaved_lo_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpacklo_pd(real, imag); }

  static simd_type interleaved_hi_from_split(simd_type const& real,
					     simd_type const& imag)
  { return _mm_unpackhi_pd(real, imag); }

  static void enter() {}
  static void exit()  {}
};
#endif
#endif

template <typename T>
struct Simd_traits<std::complex<T> > 
{
  typedef Simd_traits<T> base_traits;

  static unsigned int  const scalar_pos = 0;

  typedef typename Simd_traits<T>::simd_type base_simd_type;

  typedef std::complex<T> value_type;
  struct simd_type
  {
    base_simd_type r;
    base_simd_type i;
  };
   
  static unsigned int const vec_size = Simd_traits<T>::vec_size;
  static bool const is_accel  = Simd_traits<T>::is_accel;
  static bool const has_perm  = false;
  static bool const has_div   = Simd_traits<T>::has_div;
  static unsigned int  const alignment = Simd_traits<T>::alignment;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  {
    simd_type t = { Simd_traits<T>::zero(), Simd_traits<T>::zero() };
    return t;
  }

  static simd_type load_unaligned(value_type const* addr)
  {
    base_simd_type v0=Simd_traits<T>::load_unaligned(((T const*)addr)+0);
    base_simd_type v1=Simd_traits<T>::load_unaligned(((T const*)addr)+vec_size);
#if __ghs__
    simd_type t;
    t.r = Simd_traits<T>::real_from_interleaved(v0, v1);
    t.i = Simd_traits<T>::imag_from_interleaved(v0, v1);
#else
    // 070509: This causes an internal error with GHS:
    //         "type-change_constant: integer to bad type"
    simd_type t = 
      {
	Simd_traits<T>::real_from_interleaved(v0, v1),
	Simd_traits<T>::imag_from_interleaved(v0, v1)
      };
#endif
    return t;
  }

  static simd_type load(value_type const* addr)
  {
    base_simd_type v0 = Simd_traits<T>::load(((T const*)addr)+0);
    base_simd_type v1 = Simd_traits<T>::load(((T const*)addr)+vec_size);
#if __ghs__
    simd_type t;
    t.r = Simd_traits<T>::real_from_interleaved(v0, v1);
    t.i = Simd_traits<T>::imag_from_interleaved(v0, v1);
#else
    // 070509: This causes an internal error with GHS:
    //         "type-change_constant: integer to bad type"
    simd_type t = 
      {
	Simd_traits<T>::real_from_interleaved(v0, v1),
	Simd_traits<T>::imag_from_interleaved(v0, v1)
      };
#endif
    return t;
  }

  static simd_type load_scalar(value_type value)
  {
#if __ghs__
    simd_type t;
    t.r = Simd_traits<T>::load_scalar(value.real());
    t.i = Simd_traits<T>::load_scalar(value.imag());
#else
    simd_type t =
      {
	Simd_traits<T>::load_scalar(value.real()),
	Simd_traits<T>::load_scalar(value.imag())
      };
#endif
    return t;
  }

  static simd_type load_scalar_all(value_type value)
  {
#if __ghs__
    simd_type t;
    t.r = Simd_traits<T>::load_scalar_all(value.real());
    t.i = Simd_traits<T>::load_scalar_all(value.imag());
#else
    simd_type t =
      {
	Simd_traits<T>::load_scalar_all(value.real()),
	Simd_traits<T>::load_scalar_all(value.imag())
      };
#endif
    return t;
  }

  static void store(value_type* addr, simd_type const& vec)
  {
    base_simd_type v0 = Simd_traits<T>::interleaved_lo_from_split(vec.r,vec.i);
    base_simd_type v1 = Simd_traits<T>::interleaved_hi_from_split(vec.r,vec.i);
    Simd_traits<T>::store(((T*)addr)+0,        v0);
    Simd_traits<T>::store(((T*)addr)+vec_size, v1);
  }

  static void store_stream(value_type* addr, simd_type const& vec)
  { store(addr, vec); }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  {
#if __ghs__
    simd_type t;
    t.r = Simd_traits<T>::add(v1.r, v2.r);
    t.i = Simd_traits<T>::add(v1.i, v2.i);
#else
    simd_type t =
      {
	Simd_traits<T>::add(v1.r, v2.r),
	Simd_traits<T>::add(v1.i, v2.i)
      };
#endif
    return t;
  }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  {
#if __ghs__
    simd_type t;
    t.r = Simd_traits<T>::sub(v1.r, v2.r);
    t.i = Simd_traits<T>::sub(v1.i, v2.i);
#else
    simd_type t =
      {
	Simd_traits<T>::sub(v1.r, v2.r),
	Simd_traits<T>::sub(v1.i, v2.i)
      };
#endif
    return t;
  }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  {
    base_simd_type rr = Simd_traits<T>::mul(v1.r, v2.r);
    base_simd_type ii = Simd_traits<T>::mul(v1.i, v2.i);
    base_simd_type r  = Simd_traits<T>::sub(rr, ii);
    base_simd_type ri = Simd_traits<T>::mul(v1.r, v2.i);
    base_simd_type ir = Simd_traits<T>::mul(v1.i, v2.r);
    base_simd_type i  = Simd_traits<T>::add(ri, ir);

#if __ghs__
    simd_type t;
    t.r = r;
    t.i = i;
#else
    simd_type t = { r, i };
#endif
    return t;
  }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  {
    base_simd_type rr = Simd_traits<T>::mul(v1.r,v2.r);
    base_simd_type ii = Simd_traits<T>::mul(v1.i,v2.i);
    base_simd_type ri = Simd_traits<T>::mul(v1.i,v2.r);
    base_simd_type ir = Simd_traits<T>::mul(v1.r,v2.i);
    base_simd_type n = Simd_traits<T>::add(Simd_traits<T>::mul(v2.r,v2.r),
					   Simd_traits<T>::mul(v2.i,v2.i));
    base_simd_type r = Simd_traits<T>::div(Simd_traits<T>::add(rr,ii),n);
    base_simd_type i = Simd_traits<T>::div(Simd_traits<T>::sub(ri, ir),n);

#if __ghs__
    simd_type t;
    t.r = r;
    t.i = i;
#else
    simd_type t = { r, i };
#endif
    return t;
  }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return add(mul(v1, v2), v3);}
  // No gt.
  // No sign_mask

  static simd_type extend(simd_type const& v)
  {
    simd_type t =
      {
	Simd_traits<T>::extend(v.r),
	Simd_traits<T>::extend(v.i)
      };
    return t;
  }

  // This type *is* complex:
  //  - no real_from_interleaved
  //  - no imag_from_interleaved
  //  - no interleaved_lo_from_split
  //  - no interleaved_hi_from_split

  static void enter() { Simd_traits<T>::enter(); }
  static void exit()  { Simd_traits<T>::exit(); }
};

struct Alg_none;
struct Alg_vadd;
struct Alg_vmul;
struct Alg_rscvmul;	// (scalar real * complex vector)
struct Alg_vgt;
struct Alg_vland;
struct Alg_vlor;
struct Alg_vlxor;
struct Alg_vlnot;
struct Alg_vband;
struct Alg_vbor;
struct Alg_vbxor;
struct Alg_vbnot;
struct Alg_threshold;
struct Alg_vma_cSC;
struct Alg_vma_ip_cSC;

template <typename T,
	  bool     IsSplit,
	  typename AlgorithmTag>
struct Is_algorithm_supported
{
  static bool const value = false;
};



template <typename T>
struct Is_type_supported
{
  static bool const value = Simd_traits<T>::is_accel;
};

template <typename T>
struct Is_type_supported<std::complex<T> >
{
  static bool const value = Simd_traits<T>::is_accel;
};


} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SIMD_HPP
