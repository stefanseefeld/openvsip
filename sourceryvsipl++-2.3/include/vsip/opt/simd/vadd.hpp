/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_SIMD_VADD_HPP
#define VSIP_OPT_SIMD_VADD_HPP

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

// Define value_types for which vadd is optimized.
//  - float
//  - double
//  - complex<float>
//  - complex<double>

template <typename T,
	  bool     IsSplit>
struct Is_algorithm_supported<T, IsSplit, Alg_vadd>
{
  typedef typename Scalar_of<T>::type scalar_type;
  static bool const value =
    Simd_traits<scalar_type>::is_accel &&
    (is_same<scalar_type, float>::value || is_same<scalar_type, double>::value);
};

// Class for vadd - vector element-wise addition.
template <typename T,
	  bool     Is_vectorized>
struct Vadd;

// Generic, non-vectorized implementation of vector element-wise addition.
template <typename T>
struct Vadd<T, false>
{
  static void exec(T* A, T* B, T* R, length_type n)
  {
    while (n)
    {
      *R = *A + *B;
      R++; A++; B++;
      n--;
    }
  }
};

// Vectorized implementation of vector element-wise addition for scalars
// (float, double, etc).
template <typename T>
struct Vadd<T, true>
{
  static void exec(T* A, T* B, T* R, length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;

    // handle mis-aligned vectors
    if (simd::alignment_of(R) != simd::alignment_of(A) ||
	simd::alignment_of(R) != simd::alignment_of(B))
    {
      // PROFILE
      while (n)
      {
	*R = *A + *B;
	R++; A++; B++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of(A) != 0)
    {
      *R = *A + *B;
      R++; A++; B++;
      n--;
    }
  
    if (n == 0) return;

    simd_type reg0;
    simd_type reg1;
    simd_type reg2;
    simd_type reg3;

    simd::enter();

    while (n >= 2*simd::vec_size)
    {
      n -= 2*simd::vec_size;

      reg0 = simd::load(A);
      reg1 = simd::load(B);
      
      reg2 = simd::load(A + simd::vec_size);
      reg3 = simd::load(B + simd::vec_size);
      
      reg1 = simd::add(reg0, reg1);
      reg3 = simd::add(reg2, reg3);
      
      simd::store(R,                  reg1);
      simd::store(R + simd::vec_size, reg3);
      
      A+=2*simd::vec_size; B+=2*simd::vec_size; R+=2*simd::vec_size;
    }
    
    simd::exit();

    while (n)
    {
      *R = *A + *B;
      R++; A++; B++;
      n--;
    }
  }
};

// Vectorized implementation of vector element-wise addition for
// interleaved complex (complex<float>, complex<double>, etc).
template <typename T>
struct Vadd<std::complex<T>, true>
{
  static void exec(std::complex<T>* A, std::complex<T>* B, std::complex<T>* R,
		   length_type n)
  {
    Vadd<T, true>::exec(reinterpret_cast<T*>(A),
			reinterpret_cast<T*>(B),
			reinterpret_cast<T*>(R),
			2*n);
  }
};

// Generic, non-vectorized implementation of vector element-wise addition for
// split complex (as represented by pair<float*, float*>, etc).
template <typename T>
struct Vadd<std::pair<T, T>, false>
{
  static void exec(std::pair<T*, T*> const& A,
		   std::pair<T*, T*> const& B,
		   std::pair<T*, T*> const& R,
		   length_type n)
  {
    T const* pAr = A.first;
    T const* pAi = A.second;

    T const* pBr = B.first;
    T const* pBi = B.second;

    T* pRr = R.first;
    T* pRi = R.second;

    while (n)
    {
      *pRr = *pAr + *pBr;
      *pRi = *pAi + *pBi;
      pRr++; pRi++;
      pAr++; pAi++;
      pBr++; pBi++;
      n--;
    }
  }
};

// Vectorized implementation of vector element-wise addition for
// split complex (as represented by pair<float*, float*>, etc).
template <typename T>
struct Vadd<std::pair<T, T>, true>
{
  static void exec(std::pair<T*, T*> const& A,
		   std::pair<T*, T*> const& B,
		   std::pair<T*, T*> const& R,
		   length_type n)
  {
    Vadd<T, true>::exec(A.first, B.first, R.first, n);
    Vadd<T, true>::exec(A.second, B.second, R.second, n);
  }
};

// Depending on VSIP_IMPL_LIBSIMD_INLINE macro, either provide these
// functions inline, or provide non-inline functions in the libvsip.a.

#if VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
inline void
vadd(T *op1, T *op2, T *res, length_type size)
{
  static bool const Is_vectorized = Is_algorithm_supported<T, false, Alg_vadd>
                                      ::value;
  Vadd<T, Is_vectorized>::exec(op1, op2, res, size);
}

template <typename T>
inline void
vadd(std::pair<T*,T*>  op1, std::pair<T*,T*>  op2, std::pair<T*,T*>  res,
     length_type size)
{
  static bool const Is_vectorized = Is_algorithm_supported<T, true, Alg_vadd>
                                      ::value;
  Vadd<std::pair<T,T>, Is_vectorized>::exec(op1, op2, res, size);
}

#else

template <typename T>
void
vadd(T *op1, T *op2, T *res, length_type size);

template <typename T>
void
vadd(std::pair<T*,T*>  op1, std::pair<T*,T*>  op2, std::pair<T*,T*>  res,
     length_type size);

#endif // VSIP_IMPL_INLINE_LIBSIMD


} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SIMD_VMUL_HPP
