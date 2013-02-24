/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/svmul.hpp
    @author  Jules Bergmann
    @date    2006-03-28
    @brief   VSIPL++ Library: SIMD element-wise scalar-vector multiplication.
*/

#ifndef VSIP_OPT_SIMD_SVMUL_HPP
#define VSIP_OPT_SIMD_SVMUL_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <complex>
#include <vsip/support.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/core/metaprogramming.hpp>


namespace vsip
{
namespace impl
{
namespace simd
{


// Define value_types for which svmul is optimized.
//  - complex<float>
//  - complex<double>

template <typename T,
	  bool     IsSplit>
struct is_algorithm_supported<T, IsSplit, Alg_svmul>
{
  static bool const value =
    Simd_traits<T>::is_accel &&
    (is_same<T, float>::value || is_same<T, double>::value);
};

template <typename T,
	  bool     Is_vectorized>
struct Svmul;

// Generic, non-vectorized implementation of vector element-wise multiply.
template <typename T>
struct Svmul<T, false>
{
  // real * real
  static void exec(T alpha, T const *B, T* R, length_type n)
  {
    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }

  // complex * real (interleaved)
  static void exec(std::complex<T> alpha, T const* B, std::complex<T>* R, length_type n)
  {
    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }

  // complex * real (split)
  static void exec(std::complex<T> alpha, T const* B, 
                   std::pair<T*, T*> R, 
                   length_type n)
  {
    T* pRr = R.first;
    T* pRi = R.second;

    while (n)
    {
      *pRr = alpha.real() * *B;
      *pRi = alpha.imag() * *B;
      pRr++; pRi++;
      B++;
      n--;
    }
  }

  // complex * complex (split)
  static void exec(std::complex<T> const alpha, 
                   std::pair<T const*, T const*> const &B,
		   std::pair<T*, T*> const& R,
		   length_type n)
  {
    T const* pBr = B.first;
    T const* pBi = B.second;

    T* pRr = R.first;
    T* pRi = R.second;

    while (n)
    {
      T tmpr;
      tmpr = alpha.real() * *pBr - alpha.imag() * *pBi;
      *pRi = alpha.real() * *pBi + alpha.imag() * *pBr;
      *pRr = tmpr;
      pRr++; pRi++;
      pBr++; pBi++;
      n--;
    }
  }
};


template <typename T>
struct Svmul<std::complex<T>, false>
{
  // real * complex (interleaved)
  static void exec(T alpha, std::complex<T> const *B, std::complex<T>* R, length_type n)
  {
    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }

  // complex * complex (interleaved)
  static void exec(std::complex<T> alpha, std::complex<T> const *B, std::complex<T>* R, length_type n)
  {
    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }
};

// Generic, non-vectorized implementation of vector element-wise multiply for
// split complex (as represented by pair<float*, float*>, etc).
template <typename T>
struct Svmul<std::pair<T, T>, false>
{
  // real * complex (split)
  static void exec(T alpha, std::pair<T const*, T const*> const &B,
		   std::pair<T*, T*> const &R,
		   length_type n)
  {
    T const* pBr = B.first;
    T const* pBi = B.second;

    T* pRr = R.first;
    T* pRi = R.second;

    while (n)
    {
      *pRr = alpha * *pBr;
      *pRi = alpha * *pBi;
      pRr++; pRi++;
      pBr++; pBi++;
      n--;
    }
  }

};



// Vectorized implementation of vector element-wise multiply for
// real values (float or double)
template <typename T>
struct Svmul<T, true>
{
  // real * real
  static void exec(T alpha, T const *B, T* R, length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    // handle mis-aligned vectors
    if (simd::alignment_of((T*)R) != simd::alignment_of((T*)B))
    {
      // PROFILE
      while (n)
      {
	*R = alpha * *B;
	R++; B++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of((T*)B) != 0)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regA = simd::load_scalar_all(alpha);

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type regB = simd::load((T*)B);
      simd_type regR = simd::mul(regA, regB);
      
      simd::store((T*)R, regR);
      
      B+=simd::vec_size; R+=simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }

  // complex * real (interleaved)
  static void exec(std::complex<T> alpha, T const* B, std::complex<T>* R, length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    // handle mis-aligned vectors
    if (simd::alignment_of((T*)R) != simd::alignment_of((T*)B))
    {
      // PROFILE
      while (n)
      {
	*R = alpha * *B;
	R++; B++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of((T*)B) != 0)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regAr = simd::load_scalar_all(alpha.real());
    simd_type regAi = simd::load_scalar_all(alpha.imag());

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type regB = simd::load((T*)B);

      simd_type tmpRr = simd::mul(regAr, regB);
      simd_type tmpRi = simd::mul(regAi, regB);
      
      simd_type regR1 = simd::interleaved_lo_from_split(tmpRr, tmpRi);
      simd_type regR2 = simd::interleaved_hi_from_split(tmpRr, tmpRi);
    
      simd::store((T*)R,                  regR1);
      simd::store((T*)R + simd::vec_size, regR2);
      
      B+=simd::vec_size; R+=simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }


  // complex * real (split)
  static void exec(std::complex<T> alpha, T const* B, 
                   std::pair<T*, T*> R, 
                   length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    T* pRr = R.first;
    T* pRi = R.second;

    // handle mis-aligned vectors
    if (simd::alignment_of(B) != simd::alignment_of(pRr) ||
        simd::alignment_of(B) != simd::alignment_of(pRi))
    {
      while (n)
      {
        *pRr = alpha.real() * *B;
        *pRi = alpha.imag() * *B;
        pRr++; pRi++;
        B++;
        n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of(B) != 0)
    {
      *pRr = alpha.real() * *B;
      *pRi = alpha.imag() * *B;
      pRr++; pRi++;
      B++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regAr = simd::load_scalar_all(alpha.real());
    simd_type regAi = simd::load_scalar_all(alpha.imag());

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type regB = simd::load((T*)B);
      
      simd_type Rr = simd::mul(regAr, regB);
      simd_type Ri = simd::mul(regAi, regB);
      
      simd::store_stream(pRr, Rr);
      simd::store_stream(pRi, Ri);
      
      pRr += simd::vec_size; pRi += simd::vec_size;
      B+=simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      *pRr = alpha.real() * *B;
      *pRi = alpha.imag() * *B;
      pRr++; pRi++;
      B++;
      n--;
    }
  }


  // complex * complex (split)
  static void exec(std::complex<T> const alpha, 
                   std::pair<T const*, T const*> const &B,
		   std::pair<T*, T*> const &R,
		   length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    T const* pBr = B.first;
    T const* pBi = B.second;

    T* pRr = R.first;
    T* pRi = R.second;

    // handle mis-aligned vectors
    if (simd::alignment_of(pRr) != simd::alignment_of(pRi) ||
	simd::alignment_of(pRr) != simd::alignment_of(pBr) ||
	simd::alignment_of(pRr) != simd::alignment_of(pBi))
    {
      // PROFILE
      while (n)
      {
        T tmpr;
        tmpr = alpha.real() * *pBr - alpha.imag() * *pBi;
        *pRi = alpha.real() * *pBi + alpha.imag() * *pBr;
        *pRr = tmpr;
        pRr++; pRi++;
        pBr++; pBi++;
        n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of(pRr) != 0)
    {
      T tmpr;
      tmpr = alpha.real() * *pBr - alpha.imag() * *pBi;
      *pRi = alpha.real() * *pBi + alpha.imag() * tmpr;
      *pRr = tmpr;
      pRr++; pRi++;
      pBr++; pBi++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regAr = simd::load_scalar_all(alpha.real());
    simd_type regAi = simd::load_scalar_all(alpha.imag());

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type Br = simd::load((T*)pBr);
      simd_type Bi = simd::load((T*)pBi);
      
      simd_type Rr   = simd::sub(simd::mul(regAr, Br), simd::mul(regAi, Bi));
      simd_type Ri   = simd::add(simd::mul(regAr, Bi), simd::mul(regAi, Br));

      simd::store_stream(pRr, Rr);
      simd::store_stream(pRi, Ri);
      
      pRr += simd::vec_size; pRi += simd::vec_size;
      pBr += simd::vec_size; pBi += simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      T tmpr;
      tmpr = alpha.real() * *pBr - alpha.imag() * *pBi;
      *pRi = alpha.real() * *pBi + alpha.imag() * *pBr;
      *pRr = tmpr;
      pRr++; pRi++;
      pBr++; pBi++;
      n--;
    }
  }

};


// Vectorized implementation of vector element-wise multiply for
// interleaved complex (complex<float>, complex<double>, etc).
template <typename T>
struct Svmul<std::complex<T>, true>
{
  // real * complex (interleaved)
  static void exec(T alpha, std::complex<T> const *B, std::complex<T>* R, length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    // handle mis-aligned vectors
    if (simd::alignment_of((T*)R) != simd::alignment_of((T*)B))
    {
      // PROFILE
      while (n)
      {
	*R = alpha * *B;
	R++; B++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of((T*)B) != 0)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regA = simd::load_scalar_all(alpha);

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type regB1 = simd::load((T*)B);
      simd_type regB2 = simd::load((T*)B + simd::vec_size);

      simd_type regR1 = simd::mul(regA, regB1);
      simd_type regR2 = simd::mul(regA, regB2);
      
      simd::store((T*)R,                  regR1);
      simd::store((T*)R + simd::vec_size, regR2);
      
      B+=simd::vec_size; R+=simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }

  // complex * complex (interleaved)
  static void exec(std::complex<T> alpha, std::complex<T> const *B, std::complex<T>* R, length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    // handle mis-aligned vectors
    if (simd::alignment_of((T*)R) != simd::alignment_of((T*)B))
    {
      // PROFILE
      while (n)
      {
	*R = alpha * *B;
	R++; B++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of((T*)B) != 0)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regA1 = simd::load_scalar_all(alpha.real());
    simd_type regA2 = simd::load_scalar_all(alpha.imag());

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type tmpB1 = simd::load((T*)B);
      simd_type tmpB2 = simd::load((T*)B + simd::vec_size);

      simd_type regB1 = simd::real_from_interleaved(tmpB1, tmpB2);
      simd_type regB2 = simd::imag_from_interleaved(tmpB1, tmpB2);

      simd_type tmpR1 = simd::sub(
        simd::mul(regA1, regB1), simd::mul(regA2, regB2));

      simd_type tmpR2 = simd::add(
        simd::mul(regA1, regB2), simd::mul(regA2, regB1));

      simd_type regR1 = simd::interleaved_lo_from_split(tmpR1, tmpR2);
      simd_type regR2 = simd::interleaved_hi_from_split(tmpR1, tmpR2);
    
      simd::store((T*)R,                  regR1);
      simd::store((T*)R + simd::vec_size, regR2);

      // Pointer is increased by vec_size /complex/ elements
      B += simd::vec_size;
      R += simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      *R = alpha * *B;
      R++; B++;
      n--;
    }
  }
};


// Vectorized implementation of vector element-wise multiply for
// split complex (as represented by pair<float*, float*>, etc).
template <typename T>
struct Svmul<std::pair<T, T>, true>
{
  // real * complex (split)
  static void exec(T alpha, std::pair<T const*, T const*> const &B,
		   std::pair<T*, T*> const &R,
		   length_type n)
  {
    typedef Simd_traits<T> simd;
    typedef typename simd::simd_type simd_type;
    
    T const* pBr = B.first;
    T const* pBi = B.second;

    T* pRr = R.first;
    T* pRi = R.second;

    // handle mis-aligned vectors
    if (simd::alignment_of(pRr) != simd::alignment_of(pRi) ||
	simd::alignment_of(pRr) != simd::alignment_of(pBr) ||
	simd::alignment_of(pRr) != simd::alignment_of(pBi))
    {
      // PROFILE
      while (n)
      {
	*pRr = alpha * *pBr;
	*pRi = alpha * *pBi;
	pRr++; pRi++;
	pBr++; pBi++;
	n--;
      }
      return;
    }

    // clean up initial unaligned values
    while (n && simd::alignment_of(pRr) != 0)
    {
      *pRr = alpha * *pBr;
      *pRi = alpha * *pBi;
      pRr++; pRi++;
      pBr++; pBi++;
      n--;
    }
  
    if (n == 0) return;

    simd::enter();

    simd_type regA = simd::load_scalar_all(alpha);

    while (n >= simd::vec_size)
    {
      n -= simd::vec_size;

      simd_type Br = simd::load((T*)pBr);
      simd_type Bi = simd::load((T*)pBi);
      
      simd_type Rr   = simd::mul(regA, Br);
      simd_type Ri   = simd::mul(regA, Bi);

      simd::store_stream(pRr, Rr);
      simd::store_stream(pRi, Ri);
      
      pRr += simd::vec_size; pRi += simd::vec_size;
      pBr += simd::vec_size; pBi += simd::vec_size;
    }

    simd::exit();

    while (n)
    {
      *pRr = alpha * *pBr;
      *pRi = alpha * *pBi;
      pRr++; pRi++;
      pBr++; pBi++;
      n--;
    }
  }

};




template <typename T>
void
svmul(T op1, T const* op2, T* res, length_type size);


template <typename T>
void
svmul(T op1, std::complex<T> const* op2, std::complex<T> *res, length_type size);

template <typename T>
void
svmul(std::complex<T> op1, T const* op2, std::complex<T> *res, length_type size);

template <typename T>
void
svmul(std::complex<T> op1, std::complex<T> const* op2, std::complex<T> *res, length_type size);


template <typename T>
void
svmul(std::complex<T> const op1, T const* op2, std::pair<T*, T*> res, length_type size);

template <typename T>
void
svmul(T op1, std::pair<T const*, T const*> op2, std::pair<T*, T*> res, length_type size);

template <typename T>
void
svmul(std::complex<T> const op1, std::pair<T const*, T const*> op2, std::pair<T*, T*> res, length_type size);



} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SIMD_SVMUL_HPP
