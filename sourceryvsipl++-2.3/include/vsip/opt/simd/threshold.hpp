/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_SIMD_THRESHOLD_HPP
#define VSIP_OPT_SIMD_THRESHOLD_HPP

#include <vsip/support.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/opt/simd/expr_iterator.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

// Define value_types for which threshold is optimized.
//  - float
//  - double

template <typename T,
	  bool     IsSplit>
struct Is_algorithm_supported<T, IsSplit, Alg_threshold>
{
  static bool const value =
    Simd_traits<T>::is_accel &&
    (is_same<T, float>::value || is_same<T, double>::value);
};

// Class for threshold

// O is the binary operator
template <typename T,
          template <typename,typename> class O,
	  bool     Is_vectorized>
struct Threshold;

// Simd function to do threshold only when K is 0
// O is the binary operator
template <template <typename,typename> class O,
          typename T>
static inline length_type
thresh0(T* Z, T const* A, T const* B, length_type n)
{
  typedef Simd_traits<T>                         simd;
  typedef Simd_traits<int>                       simdi;
  typedef typename simd::simd_type               simd_type;
  typedef typename simdi::simd_type              simd_itype;
  typedef Binary_operator_map<T,O>               bin_op;

  // make sure A,B,Z are same alignment
  assert(simd::alignment_of(A) == simd::alignment_of(B) &&
	 simd::alignment_of(Z) == simd::alignment_of(A));

  simd::enter();

  while (n >= simd::vec_size)
  {
    n -= simd::vec_size;

    simd_type A_v    = simd::load(A);
    simd_type B_v    = simd::load(B);
    simd_itype mask  = simd_itype(bin_op::apply(A_v,B_v));
    simd_itype res   = simdi::band(simd_itype(A_v),mask);
    simd::store(Z,simd_type(res));

    A += simd::vec_size;
    B += simd::vec_size;
    Z += simd::vec_size;
  }

  simd::exit();

  return n;
}

// Simd function to do threshold only when K is not 0
// O is the binary operator
template <template <typename,typename> class O,
          typename T>
static inline length_type
thresh(T* Z, T const* A, T const* B, T const k, length_type n)
{
  typedef Simd_traits<T>                         simd;
  typedef Simd_traits<int>                       simdi;
  typedef typename simd::simd_type               simd_type;
  typedef typename simdi::simd_type              simd_itype;
  typedef Binary_operator_map<T,O>               bin_op;

  // make sure A,B,Z are same alignment
  assert(simd::alignment_of(A) == simd::alignment_of(B) &&
	 simd::alignment_of(Z) == simd::alignment_of(A));

  simd::enter();

  simd_type k_v    = simd::load_scalar_all(k);

  while (n >= simd::vec_size)
  {
    n -= simd::vec_size;

    simd_type A_v      = simd::load(A);
    simd_type B_v      = simd::load(B);
    simd_itype mask    = simd_itype(bin_op::apply(A_v,B_v));
    simd_itype nmask   = simdi::bnot(mask);
    simd_itype xor_val = simdi::bxor(simd_itype(A_v),simd_itype(k_v));
    simd_itype and_val = simdi::band(xor_val,nmask);
    simd_itype res     = simdi::bxor(simd_itype(A_v),and_val);
    simd::store(Z,simd_type(res));

    A += simd::vec_size;
    B += simd::vec_size;
    Z += simd::vec_size;
  }

  simd::exit();

  return n;
}


// Generic, non-vectorized implementation of threshold

template <typename T,
          template <typename,typename> class O>
struct Threshold<T, O, false>
{
  static void exec(T* Z, T* A, T* B, T k, length_type n)
  {
    while (n)
    {
      if(O<T,T>::apply(*A,*B)) *Z = *A;
      else *Z = k;
      A++;
      B++;
      Z++;
      n--;
    }
  }
};

// vectorized version

template <typename T,
          template <typename,typename> class O>
struct Threshold<T, O, true>
{
  static void exec(T* Z, T* A, T* B, T k, length_type n)
  {
    typedef Simd_traits<T>                         simd;
    typedef Simd_traits<int>                       simdi;
    typedef typename simd::simd_type               simd_type;
    typedef typename simdi::simd_type              simd_itype;

    // handle mis-aligned vectors
    if (simd::alignment_of(A) != simd::alignment_of(B) ||
	simd::alignment_of(Z) != simd::alignment_of(A))
    {
      Threshold<T,O,false>::exec(Z,A,B,k,n);
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of(A) != 0)
    {
      if(O<T,T>::apply(*A,*B)) *Z = *A;
      else *Z = k;
      A++;
      B++;
      Z++;
      n--;
    }
  
    if (n == 0) return;

    if(k != T(0))
      n = thresh<O>(Z,A,B,k,n);
    else
      n = thresh0<O>(Z,A,B,n);

    // handle last bits
    while(n)
    {
      if(O<T,T>::apply(*A,*B)) *Z = *A;
      else *Z = k;
      A++;
      B++;
      Z++;
      n--;
    }

  }
};

// Depending on VSIP_IMPL_LIBSIMD_INLINE macro, either provide these
// functions inline, or provide non-inline functions in the libvsip.a.

#if VSIP_IMPL_INLINE_LIBSIMD

template <template <typename,typename> class O,
          typename T>
inline void 
threshold(T* Z, T* A, T* B, T k, length_type n)
{
  static bool const Is_vectorized =
    Is_algorithm_supported<T, false, Alg_threshold>::value &&
    Binary_operator_map<T,O>::is_supported;
  Threshold<T, O, Is_vectorized>::exec(Z,A,B,k,n);
}

#else

template <template <typename,typename> class O,
          typename T>
void 
threshold(T* Z, T* A, T* B, T k, length_type n);

#endif

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif
