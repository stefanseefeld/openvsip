/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/simd_common.hpp
    @author  Jules Bergmann
    @date    2007-11-21
    @brief   VSIPL++ Library: SIMD traits class.

*/

#ifndef VSIP_OPT_SIMD_COMMON_HPP
#define VSIP_OPT_SIMD_COMMON_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif


#include <stdint.h>

namespace vsip
{
namespace impl
{
namespace simd
{

#if defined(_MC_EXEC)
typedef long intptr_t;
#endif

// -------------------------------------------------------------------- //
// Simd_traits -- traits class for SIMD types and operations
//
// Each specialization should define the following:
//
// Values:
//  - vec_size  - width of SIMD vector
//  - is_accel  - true  if specialization actually defines a SIMD Vector type
//              - false if default trait is used (simd_type == value_type)
//  - alignment - alignment required for SIMD types (in bytes).
//                (If alignment == 1, then no special alignment required).
//  - scalar_pos - the position of the scalar value if SIMD vector is
//                 written to array in memory.
//
// Types:
//  - value_type - base type (or element type) of SIMD vector
//  - simd_type  - SIMD vector type
//
// Alignment Utilities
//  - alignment_of    - returns 0 if address is aligned, returns
//                      number of bytes of misalignment otherwise.
//
// IO Operations:
//  - zero            - return "zero" SIMD vector.
//  - load            - load a SIMD vector from an address
//  - load_scalar     - load a scalar into pos 0 of SIMD vector
//  - load_scalar_all - load a scalar into all pos of SIMD vector
//  - store           - store a SIMD vector to an address
//
// Arithmetic Operations:
//  - add             - add two SIMD vectors together
//  - sub             - subtract two SIMD vectors
//  - mul             - multiply two SIMD vectors together
//  - div             - divide two SIMD vectors
//  - fma
//  - mag	      - magnitude (aka absolute value) of a SIMD vector
//
// Logic Operations:
//  - band            - bitwise-and two SIMD vectors
//  - bor             - bitwise-or two SIMD vectors
//  - bxor            - bitwise-xor two SIMD vectors
//  - bnot            - bitwise-negation of one SIMD vector
//
// Shuffle Operations
//  - extend                    - extend value in pos 0 to entire SIMD vector.
//  - real_from_interleaved     - create real SIMD from two interleaved SIMDs
//  - imag_from_interleaved     - create imag SIMD from two interleaved SIMDs
//  - interleaved_lo_from_split -
//  - interleaved_hi_from_split -
//  - pack                      - pack 2 SIMD vectors into 1, reducing range
//
// Architecture/Compiler Notes
//  - GCC support for Intel SSE is good (3.4, 4.0, 4.1 all work)
//  - GCC 3.4 is broken for AltiVec
//     - typedefs of vector types are not supported within a struct
//       (top-level typedefs work fine).
//  - GHS support for Altivec is good.
//     - peculiar about order: __vector must come first.
// -------------------------------------------------------------------- //
template <typename T>
struct Simd_traits;



// -------------------------------------------------------------------- //
// default class definition - defines value_type == simd_type
template <typename T>
struct Simd_traits {
  typedef T	value_type;
  typedef T	simd_type;
  typedef int   simd_itype;
   
  static int const  vec_size   = 1;
  static bool const is_accel   = false;
  static bool const has_perm   = false;
  static bool const has_div    = true;
  static int  const alignment  = 1;
  static unsigned int const scalar_pos = 0;

  static intptr_t alignment_of(value_type const* addr)
  { return (intptr_t)addr & (alignment - 1); }

  static simd_type zero()
  { return simd_type(0); }

  static simd_type load(value_type const* addr)
  { return *addr; }

  static simd_type load_unaligned(value_type const* addr)
  { return *addr; }

  static simd_type load_scalar(value_type value)
  { return value; }

  static simd_type load_scalar_all(value_type value)
  { return value; }

  static void store(value_type* addr, simd_type const& vec)
  { *addr = vec; }

  static simd_type add(simd_type const& v1, simd_type const& v2)
  { return v1 + v2; }

  static simd_type sub(simd_type const& v1, simd_type const& v2)
  { return v1 - v2; }

  static simd_type mul(simd_type const& v1, simd_type const& v2)
  { return v1 * v2; }

  static simd_type div(simd_type const& v1, simd_type const& v2)
  { return v1 / v2; }

  static simd_type mag(simd_type const& v1)
  { return mag(v1); }

  static simd_type fma(simd_type const& v1, simd_type const& v2,
		       simd_type const& v3)
  { return v1 * v2 + v3; }

  static simd_type min(simd_type const& v1, simd_type const& v2)
  { return (v1 < v2) ? v1 : v2; }

  static simd_type max(simd_type const& v1, simd_type const& v2)
  { return (v1 > v2) ? v1 : v2; }

  // These functions return ints and operate on ints
  static simd_itype band(simd_itype const& v1, simd_itype const& v2)
  { return v1 & v2; }

  static simd_itype bor(simd_itype const& v1, simd_itype const& v2)
  { return v1 | v2; }

  static simd_itype bxor(simd_itype const& v1, simd_itype const& v2)
  { return v1 ^ v2; }

  static simd_itype bnot(simd_itype const& v1)
  { return ~v1; }

  // These functions take floats and return ints
  static simd_itype gt(simd_type const& v1, simd_type const& v2)
  { return (v1 > v2) ? simd_itype(-1) : simd_itype(0); }

  static simd_itype lt(simd_type const& v1, simd_type const& v2)
  { return (v1 < v2) ? simd_itype(-1) : simd_itype(0); }

  static simd_itype ge(simd_type const& v1, simd_type const& v2)
  { return (v1 >= v2) ? simd_itype(-1) : simd_itype(0); }

  static simd_itype le(simd_type const& v1, simd_type const& v2)
  { return (v1 <= v2) ? simd_itype(-1) : simd_itype(0); }

  static simd_type pack(simd_type const&, simd_type const&)
  { assert(0); }

  static void enter() {}
  static void exit()  {}
};

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_SIMD_COMMON_HPP
