/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_SIMD_VGT_HPP
#define VSIP_OPT_SIMD_VGT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <complex>
#include <vsip/support.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/core/metaprogramming.hpp>

#define VSIP_IMPL_INLINE_LIBSIMD 0

namespace vsip
{
namespace impl
{
namespace simd
{

// Define value_types for which vgt is optimized.
//  - float

template <typename T,
	  bool     IsSplit>
struct is_algorithm_supported<T, IsSplit, Alg_vgt>
{
  static bool const value =
    Simd_traits<T>::is_accel && is_same<T, float>::value;
};



// Class for vgt - vector element-wise greater-than.

template <typename T,
	  bool     Is_vectorized>
struct Vgt;



// Generic, non-vectorized implementation of vector element-wise greater-than.

template <typename T>
struct Vgt<T, false>
{
  static void exec(T const* A, T const* B, bool* R, length_type n)
  {
    while (n)
    {
      *R = *A > *B;
      R++; A++; B++;
      n--;
    }
  }
};



// Vectorized implementation of vector element-wise gt.

// Works under the following combinations:
//  - mcoe  ppc    altivec GHS,     T=float (060728)
//  - linux x86_64 sse     GCC 3.4, T=float (060728)

template <>
struct Vgt<float, true>
{
  typedef float T;
  static void exec(T const* A, T const* B, bool* R, length_type n)
  {
    typedef vsip::impl::simd::Simd_traits<float> simd;
    typedef simd::simd_type                      simd_type;

    typedef vsip::impl::simd::Simd_traits<short> short_simd;
    typedef short_simd::simd_type                short_simd_type;

  // handle mis-aligned vectors
  if (simd::alignment_of((T*)R) != simd::alignment_of(A) ||
      simd::alignment_of((T*)R) != simd::alignment_of(B))
  {
    // PROFILE
    while (n)
    {
      *R = *A > *B;
      R++; A++; B++;
      n--;
    }
    return;
  }

  // clean up initial unaligned values
  while (n && simd::alignment_of(A) != 0)
  {
    *R = *A > *B;
    R++; A++; B++;
    n--;
  }
  
  if (n == 0) return;

  simd::enter();

  short_simd_type bool_mask =
    (short_simd_type)vsip::impl::simd::Simd_traits<signed char>::
		load_scalar_all(0x01);

  length_type const unroll = 4;
  while (n >= unroll*simd::vec_size)
  {
    n -= unroll*simd::vec_size;

    simd_type regA0 = simd::load(A);
    simd_type regA1 = simd::load(A + 1*simd::vec_size);
    simd_type regA2 = simd::load(A + 2*simd::vec_size);
    simd_type regA3 = simd::load(A + 3*simd::vec_size);
      
    simd_type regB0 = simd::load(B + 0*simd::vec_size);
    simd_type regB1 = simd::load(B + 1*simd::vec_size);
    simd_type regB2 = simd::load(B + 2*simd::vec_size);
    simd_type regB3 = simd::load(B + 3*simd::vec_size);
      
    short_simd_type cmp0 = (short_simd_type)simd::gt(regA0, regB0);
    short_simd_type cmp1 = (short_simd_type)simd::gt(regA1, regB1);
    short_simd_type cmp2 = (short_simd_type)simd::gt(regA2, regB2);
    short_simd_type cmp3 = (short_simd_type)simd::gt(regA3, regB3);

#if !__VEC__ || __BIG_ENDIAN__
    short_simd_type red0 = (short_simd_type)short_simd::pack(cmp0, cmp1);
    short_simd_type red1 = (short_simd_type)short_simd::pack(cmp2, cmp3);
    short_simd_type res  = (short_simd_type)short_simd::pack(red0, red1);
#else
    short_simd_type red0 = (short_simd_type)short_simd::pack(cmp1, cmp0);
    short_simd_type red1 = (short_simd_type)short_simd::pack(cmp3, cmp2);
    short_simd_type res  = (short_simd_type)short_simd::pack(red1, red0);
#endif

    short_simd_type bool_res = short_simd::band(res, bool_mask);

    short_simd::store((short*)R, bool_res);
      
    A += unroll*simd::vec_size;
    B += unroll*simd::vec_size;
    R += unroll*simd::vec_size;
  }
    
  simd::exit();

  while (n)
  {
    *R = *A > *B;
    R++; A++; B++;
    n--;
  }
  }
};



// Depending on VSIP_IMPL_LIBSIMD_INLINE macro, either provide these
// functions inline, or provide non-inline functions in the libvsip.a.

#if VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
inline void
vgt(T const* op1, T const* op2, bool *res, length_type size)
{
  static bool const Is_vectorized = is_algorithm_supported<T, false, Alg_vgt>
                                      ::value;
  Vgt<T, Is_vectorized>::exec(op1, op2, res, size);
}

#else

template <typename T>
void
vgt(T const* op1, T const* op2, bool *res, length_type size);

#endif // VSIP_IMPL_INLINE_LIBSIMD


} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SIMD_VMUL_HPP
