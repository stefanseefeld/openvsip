/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

#ifndef VSIP_OPT_SIMD_VMA_IP_CSC_HPP
#define VSIP_OPT_SIMD_VMA_IP_CSC_HPP

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

// Define value_types for which vma_ip is optimized.
//  - float
//  - double

template <typename T,
	  bool     IsSplit>
struct is_algorithm_supported<T, IsSplit, Alg_vma_ip_cSC>
{
  typedef typename scalar_of<T>::type scalar_type;
  static bool const value =
    Simd_traits<scalar_type>::is_accel &&
    (is_same<scalar_type, float>::value || is_same<scalar_type, double>::value);
};

// Class for vma_ip - vector element-wise multiplication.

template <typename T,
	  bool     Is_vectorized>
struct Vma_ip_cSC;

// Generic, non-vectorized implementation of vector element-wise multiply.
// C += c*S
// R += A*B 
template <typename T>
struct Vma_ip_cSC<std::complex<T>, false>
{
  static void exec(std::complex<T> const& a,
		   T const*               B,
		   std::complex<T>*       R,
		   length_type n)
  {
    while (n)
    {
      *R += a * *B;
      R++; B++;
      n--;
    }
  }
};

// Vectorized implementation of vector element-wise multiply for scalars
// (float, double, etc).
template <typename T>
struct Vma_ip_cSC<std::complex<T>, true>
{
  static void exec(std::complex<T> const& a,
		   T const*               B,
		   std::complex<T>*       R,
		   length_type n)
  {
    typedef Simd_traits<T> simd;

    typedef typename simd::simd_type simd_type;

    // handle mis-aligned vectors
    if (simd::alignment_of((T*)R) != simd::alignment_of(B))
    {
      // PROFILE
      while (n)
      {
	*R += a * *B;
	R++; B++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of((T*)R) != 0)
    {
      *R += a * *B;
      R++; B++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();


#if 0
    simd_type reg_Ar = simd::load_scalar_all(a.real());
    simd_type reg_Ai = simd::load_scalar_all(a.imag());

    while (n >= 1*simd::vec_size)
    {
      simd_type reg_B = simd::load((T*)B);

      simd_type reg_C0 = simd::load((T*)R);
      simd_type reg_C1 = simd::load((T*)R + simd::vec_size);

      simd_type reg_ABr = simd::mul(reg_Ar, reg_B);
      simd_type reg_ABi = simd::mul(reg_Ai, reg_B);

      simd_type reg_Cr = simd::real_from_interleaved(reg_C0, reg_C1);
      simd_type reg_Ci = simd::imag_from_interleaved(reg_C0, reg_C1);

      simd_type reg_Rr = simd::add(reg_Cr, reg_ABr);
      simd_type reg_Ri = simd::add(reg_Ci, reg_ABi);

      simd_type reg_R1 = simd::interleaved_lo_from_split(reg_Rr, reg_Ri);
      simd_type reg_R2 = simd::interleaved_hi_from_split(reg_Rr, reg_Ri);
      
      simd::store((T*)R,                  reg_R1);
      simd::store((T*)R + simd::vec_size, reg_R2);
      
      B += simd::vec_size; C += simd::vec_size; R += simd::vec_size;
      n -= 1*simd::vec_size;
    }
#elif 0
    simd_type reg_Ar = simd::load_scalar_all(a.real());
    simd_type reg_Ai = simd::load_scalar_all(a.imag());
    simd_type reg_A  = simd::interleaved_lo_from_split(reg_Ar, reg_Ai);

    while (n >= 1*simd::vec_size)
    {
      simd_type reg_B = simd::load((T*)B);

      simd_type reg_C1 = simd::load((T*)R);
      simd_type reg_C2 = simd::load((T*)R + simd::vec_size);

      simd_type reg_B1 = simd::interleaved_lo_from_split(reg_B, reg_B);
      simd_type reg_B2 = simd::interleaved_hi_from_split(reg_B, reg_B);

      simd_type reg_AB1 = simd::mul(reg_A, reg_B1);
      simd_type reg_AB2 = simd::mul(reg_A, reg_B2);

      simd_type reg_R1 = simd::add(reg_C1, reg_AB1);
      simd_type reg_R2 = simd::add(reg_C2, reg_AB2);

      simd::store((T*)R,                  reg_R1);
      simd::store((T*)R + simd::vec_size, reg_R2);
      
      B += simd::vec_size; C += simd::vec_size; R += simd::vec_size;
      n -= 1*simd::vec_size;
    }
#else
    simd_type reg_Ar = simd::load_scalar_all(a.real());
    simd_type reg_Ai = simd::load_scalar_all(a.imag());
    simd_type reg_A  = simd::interleaved_lo_from_split(reg_Ar, reg_Ai);

    while (n >= 2*simd::vec_size)
    {
      simd_type reg_B01 = simd::load((T*)B);
      simd_type reg_B23 = simd::load((T*)B + simd::vec_size);

      simd_type reg_C0 = simd::load((T*)R);
      simd_type reg_C1 = simd::load((T*)R + simd::vec_size);
      simd_type reg_C2 = simd::load((T*)R + 2*simd::vec_size);
      simd_type reg_C3 = simd::load((T*)R + 3*simd::vec_size);

      simd_type reg_B0 = simd::interleaved_lo_from_split(reg_B01, reg_B01);
      simd_type reg_B1 = simd::interleaved_hi_from_split(reg_B01, reg_B01);
      simd_type reg_B2 = simd::interleaved_hi_from_split(reg_B23, reg_B23);
      simd_type reg_B3 = simd::interleaved_hi_from_split(reg_B23, reg_B23);

      simd_type reg_AB0 = simd::mul(reg_A, reg_B0);
      simd_type reg_AB1 = simd::mul(reg_A, reg_B1);
      simd_type reg_AB2 = simd::mul(reg_A, reg_B2);
      simd_type reg_AB3 = simd::mul(reg_A, reg_B3);

      simd_type reg_R0 = simd::add(reg_C0, reg_AB0);
      simd_type reg_R1 = simd::add(reg_C1, reg_AB1);
      simd_type reg_R2 = simd::add(reg_C2, reg_AB2);
      simd_type reg_R3 = simd::add(reg_C3, reg_AB3);

      simd::store((T*)R,                    reg_R0);
      simd::store((T*)R + 1*simd::vec_size, reg_R1);
      simd::store((T*)R + 2*simd::vec_size, reg_R2);
      simd::store((T*)R + 3*simd::vec_size, reg_R3);
      
      B += 2*simd::vec_size; R += 2*simd::vec_size;
      n -= 2*simd::vec_size;
    }
#endif
    
    simd::exit();

    while (n)
    {
      *R += a * *B;
      R++; B++;
      n--;
    }
  }
};



// Depending on VSIP_IMPL_LIBSIMD_INLINE macro, either provide these
// functions inline, or provide non-inline functions in the libvsip.a.

#if VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
inline void
vma_ip_cSC(std::complex<T> const& a,
	   T const*               B,
	   std::complex<T>*       R,
	   length_type n)
{
  static bool const Is_vectorized =
    is_algorithm_supported<std::complex<T>, false, Alg_vma_ip_cSC>::value;
  Vma_ip_cSC<std::complex<T>, Is_vectorized>::exec(a, B, R, n);
}

#else

template <typename T>
void
vma_ip_cSC(std::complex<T> const& a,
	   T const*               B,
	   std::complex<T>*       R,
	   length_type n);

#endif // VSIP_IMPL_INLINE_LIBSIMD


} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SIMD_VMA_IP_CSC_HPP
