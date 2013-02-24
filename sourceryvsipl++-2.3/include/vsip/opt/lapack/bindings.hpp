/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/lapack/bindings.hpp
    @author  Jules Bergmann
    @date    2005-08-19
    @brief   VSIPL++ Library: Lapack interface

NOTES:
 [0] LAPACK is a Fortran API.  There is not a standard C API, as there is
     with BLAS.  Some LAPACK packages do not provide C headers (for
     instance lapack3 on debian), while other do (for instance MKL).

 [1] Passing std::complex<float> as T to the macros VSIPL_IMPL_LAPACK_GEQRF,
     etc. triggers a pre-processor bug for Intel C++.  'geqrf_blksize<T>'
     gets expanded to 'geqrf_blksize<complex<float>>'.  As a result ICC
     complains about the missing space in nested template argument lists.

     To avoid this, we add an extra space after T in the macro:
        'geqrf_blksize<T >'
*/

#ifndef VSIP_OPT_LAPACK_MISC_HPP
#define VSIP_OPT_LAPACK_MISC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <complex>

#include <vsip/support.hpp>
#include <vsip/core/config.hpp>
#include <vsip/core/metaprogramming.hpp>

#ifdef VSIP_IMPL_USE_F2C_ABI
# define FORTRAN_FLOAT_RETURN double
# define USE_F2C_COMPLEX_RETURN 1
#else
# define FORTRAN_FLOAT_RETURN float
# define USE_F2C_COMPLEX_RETURN 0
#endif


#define CVT_TRANSPOSE(c) \
   (((c) == 'N' || (c) == 'n') ? CblasNoTrans : \
    ((c) == 'T' || (c) == 't') ? CblasTrans : \
    ((c) == 'C' || (c) == 'c') ? CblasConjTrans : \
    CblasNoTrans)

#define CVT_UPLO(c) \
   (((c) == 'U' || (c) == 'u') ? CblasUpper : \
    ((c) == 'L' || (c) == 'l') ? CblasLower : \
    CblasLower)

#define CVT_DIAG(c) \
   (((c) == 'U' || (c) == 'u') ? CblasUnit : \
    ((c) == 'N' || (c) == 'n') ? CblasNonUnit : \
    CblasNonUnit)

#define CVT_SIDE(c) \
   (((c) == 'L' || (c) == 'l') ? CblasLeft : \
    ((c) == 'R' || (c) == 'r') ? CblasRight : \
    CblasRight)





extern "C"
{

// Include the appropriate CBLAS header, depending on which library
// we're using.
// 
// If VSIP_IMPL_USE_CBLAS == 1, we're using ATLAS' CBLAS.
// If VSIP_IMPL_USE_CBLAS == 2, we're using MKL's CBLAS.
// If VSIP_IMPL_USE_CBLAS == 3, we're using ACML's psuedo-CBLAS.
//
// ACML doesn't provide a CBLAS API.  However, it does provide C
// linkage to BLAS functions.
//  - For the dot-product routines that have a non-void return type, We
//    use our own CBLAS wrappers on top of the ACML C linkage to avoid
//    potential ABI issues (VISP_IMPL_USE_CBLAS_DOT == 1).
//  - For other BLAS routines, we use the ACML Fortran linkage.


#if VSIP_IMPL_USE_CBLAS == 1
#  include <cblas.h>
#  define VSIP_IMPL_USE_CBLAS_DOT    1
#  define VSIP_IMPL_USE_CBLAS_OTHERS 1
#elif VSIP_IMPL_USE_CBLAS == 2
#  include <mkl_cblas.h>
#  define VSIP_IMPL_USE_CBLAS_DOT    1
#  define VSIP_IMPL_USE_CBLAS_OTHERS 1
#elif VSIP_IMPL_USE_CBLAS == 3
#  include <vsip/opt/lapack/acml_cblas.hpp>
#  define VSIP_IMPL_USE_CBLAS_DOT    1
#  define VSIP_IMPL_USE_CBLAS_OTHERS 0
#else
#  define VSIP_IMPL_USE_CBLAS_DOT    0
#  define VSIP_IMPL_USE_CBLAS_OTHERS 0
#endif
}



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace blas
{

extern "C"
{
  typedef int*                  I;
  typedef float*                S;
  typedef double*               D;
  typedef std::complex<float>*  C;
  typedef std::complex<double>* Z;

#if !VSIP_IMPL_USE_CBLAS_DOT
  // dot
  FORTRAN_FLOAT_RETURN sdot_ (I, S, I, S, I);
  double ddot_ (I, D, I, D, I);

# if USE_F2C_COMPLEX_RETURN
  void cdotu_(C, I, C, I, C, I);
  void zdotu_(Z, I, Z, I, Z, I);
  void cdotc_(C, I, C, I, C, I);
  void zdotc_(Z, I, Z, I, Z, I);
# else
  // We need to declare functions returning a 
  // C99 complex type, not a C++ complex<>.
  float _Complex cdotu_(I, C, I, C, I);
  double _Complex zdotu_(I, Z, I, Z, I);
  float _Complex cdotc_(I, C, I, C, I);
  double _Complex zdotc_(I, Z, I, Z, I);
# endif
#endif

#if !VSIP_IMPL_USE_CBLAS_OTHERS
  // trsm
  void strsm_ (char*, char*, char*, char*, I, I, S, S, I, S, I);
  void dtrsm_ (char*, char*, char*, char*, I, I, D, D, I, D, I);
  void ctrsm_ (char*, char*, char*, char*, I, I, C, C, I, C, I);
  void ztrsm_ (char*, char*, char*, char*, I, I, Z, Z, I, Z, I);

  // gemv
  void sgemv_(char*, I, I, S, S, I, S, I, S, S, I);
  void dgemv_(char*, I, I, D, D, I, D, I, D, D, I);
  void cgemv_(char*, I, I, C, C, I, C, I, C, C, I);
  void zgemv_(char*, I, I, Z, Z, I, Z, I, Z, Z, I);

  // gemm
  void sgemm_(char*, char*, I, I, I, S, S, I, S, I, S, S, I);
  void dgemm_(char*, char*, I, I, I, D, D, I, D, I, D, D, I);
  void cgemm_(char*, char*, I, I, I, C, C, I, C, I, C, C, I);
  void zgemm_(char*, char*, I, I, I, Z, Z, I, Z, I, Z, Z, I);

  // ger
  void sger_  ( I, I, S, S, I, S, I, S, I );
  void dger_  ( I, I, D, D, I, D, I, D, I );
  void cgerc_ ( I, I, C, C, I, C, I, C, I );
  void zgerc_ ( I, I, Z, Z, I, Z, I, Z, I );
  void cgeru_ ( I, I, C, C, I, C, I, C, I );
  void zgeru_ ( I, I, Z, Z, I, Z, I, Z, I );
#endif
};

#define VSIP_IMPL_CBLAS_DOT(T, VPPFCN, BLASFCN)				\
inline T								\
VPPFCN(int n,								\
    T* x, int incx,							\
    T* y, int incy)							\
{									\
  if (incx < 0) x += incx * (n-1);                                      \
  if (incy < 0) y += incy * (n-1);                                      \
  return BLASFCN(n, x, incx, y, incy);				\
}

#define VSIP_IMPL_BLAS_DOT(T, VPPFCN, BLASFCN)				\
inline T								\
VPPFCN(int n,								\
    T* x, int incx,							\
    T* y, int incy)							\
{									\
  if (incx < 0) x += incx * (n-1);                                      \
  if (incy < 0) y += incy * (n-1);                                      \
  return BLASFCN(&n, x, &incx, y, &incy);				\
}

#if VSIP_IMPL_USE_CBLAS_DOT
  VSIP_IMPL_CBLAS_DOT(float,                dot, cblas_sdot)
  VSIP_IMPL_CBLAS_DOT(double,               dot, cblas_ddot)
#else
  VSIP_IMPL_BLAS_DOT(float,                dot, sdot_)
  VSIP_IMPL_BLAS_DOT(double,               dot, ddot_)
#endif // VSIP_IMPL_USE_CBLAS_DOT

#undef VSIP_IMPL_BLAS_DOT



#define VSIP_IMPL_CBLAS_CDOT(T, VPPFCN, BLASFCN)			\
inline std::complex<T>	       						\
VPPFCN(int n,                                                           \
       std::complex<T>* x, int incx,                                    \
       std::complex<T>* y, int incy)                                    \
{									\
  std::complex<T> ret;  		       				\
  if (incx < 0) x += incx * (n-1);                                      \
  if (incy < 0) y += incy * (n-1);                                      \
  BLASFCN(n,								\
	  static_cast<const void*>(x), incx,				\
	  static_cast<const void*>(y), incy, &ret);			\
  return ret;                                                           \
}

# if USE_F2C_COMPLEX_RETURN
# define VSIP_IMPL_BLAS_CDOT(T, VPPFCN, BLASFCN)			\
inline std::complex<T>                                                  \
VPPFCN(int n,                                                           \
       std::complex<T>* x, int incx,                                    \
       std::complex<T>* y, int incy)                                    \
{									\
  std::complex<T> ret;							\
  if (incx < 0) x += incx * (n-1);                                      \
  if (incy < 0) y += incy * (n-1);                                      \
  BLASFCN(&ret, &n, x, &incx, y, &incy);				\
  return ret;								\
}
#else
# define VSIP_IMPL_BLAS_CDOT(T, VPPFCN, BLASFCN)			\
inline std::complex<T>                                                  \
VPPFCN(int n,                                                           \
       std::complex<T>* x, int incx,                                    \
       std::complex<T>* y, int incy)                                    \
{									\
  if (incx < 0) x += incx * (n-1);                                      \
  if (incy < 0) y += incy * (n-1);                                      \
  T __complex__ retn = BLASFCN(&n, x, &incx, y, &incy);			\
  return std::complex<T>(__real__ retn, __imag__ retn);                 \
}
#endif

#if VSIP_IMPL_USE_CBLAS_DOT
  VSIP_IMPL_CBLAS_CDOT(float,  dot, cblas_cdotu_sub)
  VSIP_IMPL_CBLAS_CDOT(double, dot, cblas_zdotu_sub)

  VSIP_IMPL_CBLAS_CDOT(float,  dotc, cblas_cdotc_sub)
  VSIP_IMPL_CBLAS_CDOT(double, dotc, cblas_zdotc_sub)
#else
  VSIP_IMPL_BLAS_CDOT(float,  dot, cdotu_)
  VSIP_IMPL_BLAS_CDOT(double, dot, zdotu_)

  VSIP_IMPL_BLAS_CDOT(float,  dotc, cdotc_)
  VSIP_IMPL_BLAS_CDOT(double, dotc, zdotc_)
#endif

#undef VSIP_IMPL_BLAS_CDOT



#define VSIP_IMPL_BLAS_TRSM(T, FCN)					\
inline void								\
trsm(char side, char uplo,char transa,char diag,			\
     int m, int n,							\
     T alpha,								\
     T *a, int lda,							\
     T *b, int ldb)							\
{									\
  FCN(&side, &uplo, &transa, &diag,					\
      &m, &n,								\
      &alpha,								\
      a, &lda,								\
      b, &ldb);								\
}

/* Define an overloaded function that helps us pas some scalars to the cblas
 * functions. Some cblas functions require the argument to be passed as a
 * pointer when it is complex but by reference otherwise. This makes the
 * defines a little easier to look at.
 */

inline float  cblas_scalar_cast(float arg) { return arg; }
inline double cblas_scalar_cast(double arg) { return arg; }
inline void*  cblas_scalar_cast(std::complex<float> &arg)
  { return static_cast<void*> (&arg); }
inline void*  cblas_scalar_cast(std::complex<double> &arg)
  { return static_cast<void*> (&arg); }


#define VSIP_IMPL_CBLAS_TRSM(T, FCN)					\
inline void								\
trsm(char side, char uplo,char transa,char diag,			\
     int m, int n,							\
     T alpha,								\
     T *a, int lda,							\
     T *b, int ldb)							\
{									\
  FCN(CblasColMajor,							\
      CVT_SIDE(side), CVT_UPLO(uplo),CVT_TRANSPOSE(transa),		\
      CVT_DIAG(diag),							\
      m, n,								\
      cblas_scalar_cast(alpha),						\
      a, lda,							\
      b, ldb);							\
}

#if VSIP_IMPL_USE_CBLAS_OTHERS
VSIP_IMPL_CBLAS_TRSM(float,               cblas_strsm)
VSIP_IMPL_CBLAS_TRSM(double,              cblas_dtrsm)
VSIP_IMPL_CBLAS_TRSM(std::complex<float>, cblas_ctrsm)
VSIP_IMPL_CBLAS_TRSM(std::complex<double>,cblas_ztrsm)
#else
VSIP_IMPL_BLAS_TRSM(float,                strsm_)
VSIP_IMPL_BLAS_TRSM(double,               dtrsm_)
VSIP_IMPL_BLAS_TRSM(std::complex<float>,  ctrsm_)
VSIP_IMPL_BLAS_TRSM(std::complex<double>, ztrsm_)
#endif

#undef VSIP_IMPL_CBLAS_TRSM
#undef VSIP_IMPL_BLAS_TRSM



#define VSIP_IMPL_BLAS_GEMV(T, FCN)					\
inline void								\
gemv(char transa,            						\
     int m, int n,       						\
     T alpha,								\
     T *a, int lda,							\
     T *x, int incx,							\
     T beta,								\
     T *y, int incy)							\
{									\
  FCN(&transa,            						\
      &m, &n,    							\
      &alpha,								\
      a, &lda,								\
      x, &incx,								\
      &beta,								\
      y, &incy);							\
}

#define VSIP_IMPL_CBLAS_GEMV(T, FCN)					\
inline void								\
gemv(char transa,            						\
     int m, int n,       						\
     T alpha,								\
     T *a, int lda,							\
     T *x, int incx,							\
     T beta,								\
     T *y, int incy)							\
{									\
  FCN(CblasColMajor, 							\
      CVT_TRANSPOSE(transa), 						\
      m, n,    								\
      cblas_scalar_cast(alpha),						\
      a, lda,								\
      x, incx,								\
      cblas_scalar_cast(beta),						\
      y, incy);								\
}


#if VSIP_IMPL_USE_CBLAS_OTHERS
VSIP_IMPL_CBLAS_GEMV(float,               cblas_sgemv)
VSIP_IMPL_CBLAS_GEMV(double,              cblas_dgemv)
VSIP_IMPL_CBLAS_GEMV(std::complex<float>, cblas_cgemv)
VSIP_IMPL_CBLAS_GEMV(std::complex<double>,cblas_zgemv)
#else
VSIP_IMPL_BLAS_GEMV(float,                sgemv_)
VSIP_IMPL_BLAS_GEMV(double,               dgemv_)
VSIP_IMPL_BLAS_GEMV(std::complex<float>,  cgemv_)
VSIP_IMPL_BLAS_GEMV(std::complex<double>, zgemv_)
#endif

#undef VSIP_IMPL_CBLAS_GEMV
#undef VSIP_IMPL_BLAS_GEMV



#define VSIP_IMPL_BLAS_GEMM(T, FCN)					\
inline void								\
gemm(char transa, char transb,						\
     int m, int n, int k,						\
     T alpha,								\
     T *a, int lda,							\
     T *b, int ldb,							\
     T beta,								\
     T *c, int ldc)							\
{									\
  FCN(&transa, &transb,							\
      &m, &n, &k,							\
      &alpha,								\
      a, &lda,								\
      b, &ldb,								\
      &beta,								\
      c, &ldc);								\
}

#define VSIP_IMPL_CBLAS_GEMM(T, FCN)					\
inline void								\
gemm(char transa, char transb,						\
     int m, int n, int k,						\
     T alpha,								\
     T *a, int lda,							\
     T *b, int ldb,							\
     T beta,								\
     T *c, int ldc)							\
{									\
  FCN(CblasColMajor,							\
      CVT_TRANSPOSE(transa), CVT_TRANSPOSE(transb),			\
      m, n, k,								\
      cblas_scalar_cast(alpha),						\
      a, lda,								\
      b, ldb,								\
      cblas_scalar_cast(beta),						\
      c, ldc);								\
}


#if VSIP_IMPL_USE_CBLAS_OTHERS
VSIP_IMPL_CBLAS_GEMM(float,                cblas_sgemm)
VSIP_IMPL_CBLAS_GEMM(double,               cblas_dgemm)
VSIP_IMPL_CBLAS_GEMM(std::complex<float>,  cblas_cgemm)
VSIP_IMPL_CBLAS_GEMM(std::complex<double>, cblas_zgemm)
#else
VSIP_IMPL_BLAS_GEMM(float,                sgemm_)
VSIP_IMPL_BLAS_GEMM(double,               dgemm_)
VSIP_IMPL_BLAS_GEMM(std::complex<float>,  cgemm_)
VSIP_IMPL_BLAS_GEMM(std::complex<double>, zgemm_)
#endif

#undef VSIP_IMPL_BLAS_GEMM
#undef VSIP_IMPL_CBLAS_GEMM



#define VSIP_IMPL_BLAS_GER(T, VPPFCN, FCN)     	\
inline void					\
VPPFCN( int m, int n,           		\
     T alpha,					\
     T *x, int incx,				\
     T *y, int incy,				\
     T *a, int lda)				\
{						\
  FCN(&m, &n,    				\
      &alpha,					\
      x, &incx,					\
      y, &incy,					\
      a, &lda);					\
}

#define VSIP_IMPL_CBLAS_GER(T, VPPFCN, FCN)    	\
inline void					\
VPPFCN( int m, int n,           		\
     T alpha,					\
     T *x, int incx,				\
     T *y, int incy,				\
     T *a, int lda)				\
{						\
  FCN(CblasColMajor,				\
      m, n,    					\
      cblas_scalar_cast(alpha),			\
      x, incx,					\
      y, incy,					\
      a, lda);					\
}

#if VSIP_IMPL_USE_CBLAS_OTHERS
VSIP_IMPL_CBLAS_GER(float,                ger, cblas_sger)
VSIP_IMPL_CBLAS_GER(double,               ger, cblas_dger)
VSIP_IMPL_CBLAS_GER(std::complex<float>,  gerc, cblas_cgerc)
VSIP_IMPL_CBLAS_GER(std::complex<double>, gerc, cblas_zgerc)
VSIP_IMPL_CBLAS_GER(std::complex<float>,  geru, cblas_cgeru)
VSIP_IMPL_CBLAS_GER(std::complex<double>, geru, cblas_zgeru)
#else
VSIP_IMPL_BLAS_GER(float,                ger, sger_)
VSIP_IMPL_BLAS_GER(double,               ger, dger_)
VSIP_IMPL_BLAS_GER(std::complex<float>,  gerc, cgerc_)
VSIP_IMPL_BLAS_GER(std::complex<double>, gerc, zgerc_)
VSIP_IMPL_BLAS_GER(std::complex<float>,  geru, cgeru_)
VSIP_IMPL_BLAS_GER(std::complex<double>, geru, zgeru_)
#endif

#undef VSIP_IMPL_BLAS_GER
#undef VSIP_IMPL_CBLAS_GER



template <typename T>
struct Blas_traits
{
  static bool const valid = false;
};

template <>
struct Blas_traits<float>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct Blas_traits<double>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct Blas_traits<std::complex<float> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

template <>
struct Blas_traits<std::complex<double> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

} // namespace vsip::impl::blas



/***********************************************************************
  Lapack 
***********************************************************************/

namespace lapack
{

extern "C"
{
  typedef int*                  I;
  typedef float*                S;
  typedef double*               D;
  typedef std::complex<float>*  C;
  typedef std::complex<double>* Z;

  // lapack
  void sgeqrf_(I, I, S, I, S, S, I, I);
  void dgeqrf_(I, I, D, I, D, D, I, I);
  void cgeqrf_(I, I, C, I, C, C, I, I);
  void zgeqrf_(I, I, Z, I, Z, Z, I, I);

  void sgeqr2_(I, I, S, I, S, S, I);
  void dgeqr2_(I, I, D, I, D, D, I);
  void cgeqr2_(I, I, C, I, C, C, I);
  void zgeqr2_(I, I, Z, I, Z, Z, I);

  void sormqr_(char*, char*, I, I, I, S, I, S, S, I, S, I, I);
  void dormqr_(char*, char*, I, I, I, D, I, D, D, I, D, I, I);
  void cunmqr_(char*, char*, I, I, I, C, I, C, C, I, C, I, I);
  void zunmqr_(char*, char*, I, I, I, Z, I, Z, Z, I, Z, I, I);

  void sgebrd_(I, I, S, I, S, S, S, S, S, I, I);
  void dgebrd_(I, I, D, I, D, D, D, D, D, I, I);
  void cgebrd_(I, I, C, I, S, S, C, C, C, I, I);
  void zgebrd_(I, I, Z, I, D, D, Z, Z, Z, I, I);

  void sorgbr_(char*, I, I, I, S, I, S, S, I, I);
  void dorgbr_(char*, I, I, I, D, I, D, D, I, I);
  void cungbr_(char*, I, I, I, C, I, C, C, I, I);
  void zungbr_(char*, I, I, I, Z, I, Z, Z, I, I);

  void sbdsqr_(char*, I, I, I, I, S, S, S, I, S, I, S, I, S, I);
  void dbdsqr_(char*, I, I, I, I, D, D, D, I, D, I, D, I, D, I);
  void cbdsqr_(char*, I, I, I, I, S, S, C, I, C, I, C, I, C, I);
  void zbdsqr_(char*, I, I, I, I, D, D, Z, I, Z, I, Z, I, Z, I);

  void spotrf_(char*, I, S, I, I);
  void dpotrf_(char*, I, D, I, I);
  void cpotrf_(char*, I, C, I, I);
  void zpotrf_(char*, I, Z, I, I);

  void spotrs_(char*, I, I, S, I, S, I, I);
  void dpotrs_(char*, I, I, D, I, D, I, I);
  void cpotrs_(char*, I, I, C, I, C, I, I);
  void zpotrs_(char*, I, I, Z, I, Z, I, I);

  void sgetrf_(I, I, S, I, I, I);
  void dgetrf_(I, I, D, I, I, I);
  void cgetrf_(I, I, C, I, I, I);
  void zgetrf_(I, I, Z, I, I, I);

  void sgetrs_(char*, I, I, S, I, I, S, I, I);
  void dgetrs_(char*, I, I, D, I, I, D, I, I);
  void cgetrs_(char*, I, I, C, I, I, C, I, I);
  void zgetrs_(char*, I, I, Z, I, I, Z, I, I);

#if VSIP_IMPL_USE_LAPACK_ILAENV
  int ilaenv_(I, char const *, char const *, I, I, I, I);
#endif
} // extern "C"

#if VSIP_IMPL_USE_LAPACK_ILAENV
inline int
ilaenv(int ispec, char const *name, char const *opts, int n1, int n2, int n3, int n4)
{
  return ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4);
}
#else
inline int
ilaenv(int, char const *, char const *, int , int , int , int)
{
  return 80;
}
#endif



#define VSIP_IMPL_LAPACK_GEQRF(T, FCN, NAME)				\
inline void geqrf(int m, int n, T* a, int lda, T* tau, T* work,		\
		  int& lwork)						\
{									\
  int info;								\
  FCN(&m, &n, a, &lda, tau, work, &lwork, &info);			\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::geqrf -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}									\
									\
template <>								\
inline int								\
geqrf_blksize<T >(int m, int n) /* Note [1] */				\
{									\
  return ilaenv(1, NAME, "", m, n, -1, -1);				\
}

template <typename T>
inline int
geqrf_blksize(int m, int n);


VSIP_IMPL_LAPACK_GEQRF(float,                sgeqrf_, "sgeqrf")
VSIP_IMPL_LAPACK_GEQRF(double,               dgeqrf_, "dgeqrf")
VSIP_IMPL_LAPACK_GEQRF(std::complex<float>,  cgeqrf_, "cgeqrf")
VSIP_IMPL_LAPACK_GEQRF(std::complex<double>, zgeqrf_, "zgeqrf")

#undef VSIP_IMPL_LAPACK_GEQRF



#define VSIP_IMPL_LAPACK_GEQR2(T, FCN, NAME)				\
inline void geqr2(int m, int n, T* a, int lda, T* tau, T* work,		\
		  int& /*lwork*/)					\
{									\
  int info;								\
  FCN(&m, &n, a, &lda, tau, work, &info);				\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::geqr2 -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}									\
									\
template <>								\
inline int								\
geqr2_blksize<T >(int m, int n) /* Note [1] */				\
{									\
  return ilaenv(1, NAME, "", m, n, -1, -1);				\
}

template <typename T>
inline int
geqr2_blksize(int m, int n);


VSIP_IMPL_LAPACK_GEQR2(float,                sgeqr2_, "sgeqr2")
VSIP_IMPL_LAPACK_GEQR2(double,               dgeqr2_, "dgeqr2")
VSIP_IMPL_LAPACK_GEQR2(std::complex<float>,  cgeqr2_, "cgeqr2")
VSIP_IMPL_LAPACK_GEQR2(std::complex<double>, zgeqr2_, "zgeqr2")

#undef VSIP_IMPL_LAPACK_GEQR2



#define VSIP_IMPL_LAPACK_MQR(T, FCN, NAME)				\
inline void mqr(char side, char trans,					\
		int m, int n, int k,					\
		T *a, int lda,						\
		T *tau,							\
		T *c, int ldc,						\
		T *work, int& lwork)					\
{									\
  int info;								\
  FCN(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, &info); \
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::mqr -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}									\
									\
template <>								\
inline int								\
mqr_blksize<T >(char side, char trans, int m, int n, int k) /* Note [1] */ \
{									\
  char arg[4]; arg[0] = side; arg[1] = trans; arg[2] = 0;		\
  return ilaenv(1, NAME, arg, m, n, k, -1);				\
}

template <typename T>
inline int
mqr_blksize(char side, char trans, int m, int n, int k);

VSIP_IMPL_LAPACK_MQR(float,                sormqr_, "sormqr")
VSIP_IMPL_LAPACK_MQR(double,               dormqr_, "dormqr")
VSIP_IMPL_LAPACK_MQR(std::complex<float>,  cunmqr_, "cunmqr")
VSIP_IMPL_LAPACK_MQR(std::complex<double>, zunmqr_, "zunmqr")

#undef VSIP_IMPL_LAPACK_MQR



#define VSIP_IMPL_LAPACK_GEBRD(T, FCN, NAME)				\
inline void gebrd(int m, int n, T* a, int lda,				\
		  vsip::impl::Scalar_of<T >::type* d,			\
		  vsip::impl::Scalar_of<T >::type* e,			\
		  T* tauq, T* taup, T* work, int& lwork)		\
{									\
  int info;								\
  FCN(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);		\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::gebrd -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}									\
									\
template <>								\
inline int								\
gebrd_blksize<T >(int m, int n) /* Note [1] */				\
{									\
  return ilaenv(1, NAME, "", m, n, -1, -1);				\
}

template <typename T>
inline int
gebrd_blksize(int m, int n);


VSIP_IMPL_LAPACK_GEBRD(float,                sgebrd_, "sgebrd")
VSIP_IMPL_LAPACK_GEBRD(double,               dgebrd_, "dgebrd")
VSIP_IMPL_LAPACK_GEBRD(std::complex<float>,  cgebrd_, "cgebrd")
VSIP_IMPL_LAPACK_GEBRD(std::complex<double>, zgebrd_, "zgebrd")

#undef VSIP_IMPL_LAPACK_GEBRD



#define VSIP_IMPL_LAPACK_GBR(T, FCN, NAME)				\
inline void gbr(char vect,						\
		int m, int n, int k,					\
		T *a, int lda,						\
		T *tau,							\
		T *work, int& lwork)					\
{									\
  int info;								\
  FCN(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);		\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::gbr -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}									\
									\
template <>								\
inline int								\
gbr_blksize<T >(char vect, int m, int n, int k) /* Note [1] */		\
{									\
  char arg[2]; arg[0] = vect; arg[1] = 0;				\
  return ilaenv(1, NAME, arg, m, n, k, -1);				\
}

template <typename T>
inline int
gbr_blksize(char vect, int m, int n, int k);

VSIP_IMPL_LAPACK_GBR(float,                sorgbr_, "sorgbr")
VSIP_IMPL_LAPACK_GBR(double,               dorgbr_, "dorgbr")
VSIP_IMPL_LAPACK_GBR(std::complex<float>,  cungbr_, "cungbr")
VSIP_IMPL_LAPACK_GBR(std::complex<double>, zungbr_, "zungbr")

#undef VSIP_IMPL_LAPACK_GBR



/// BDSQR - compute the singular value decomposition of a general matrix
///         that has been reduce to bidiagonal form.
#define VSIP_IMPL_LAPACK_BDSQR(T, FCN)					\
inline void bdsqr(char uplo,						\
		  int n, int ncvt, int nru, int ncc,			\
		  vsip::impl::Scalar_of<T >::type* d,			\
		  vsip::impl::Scalar_of<T >::type* e,			\
		  T *vt, int ldvt,					\
		  T *u, int ldu,					\
		  T *c, int ldc,					\
		  T *work)						\
{									\
  int info;								\
  FCN(&uplo, &n, &ncvt, &nru, &ncc, d, e,				\
      vt, &ldvt, u, &ldu, c, &ldc, work, &info);			\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::bdsqr -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}

VSIP_IMPL_LAPACK_BDSQR(float,                sbdsqr_)
VSIP_IMPL_LAPACK_BDSQR(double,               dbdsqr_)
VSIP_IMPL_LAPACK_BDSQR(std::complex<float>,  cbdsqr_)
VSIP_IMPL_LAPACK_BDSQR(std::complex<double>, zbdsqr_)

#undef VSIP_IMPL_LAPACK_BDSQR



/// POTRF - compute cholesky factorization of a symmetric (hermtian)
/// postive definite matrix
///
/// :Returns:
///   true  if info == 0
///   false if info > 0,
///   (When info > 0, this indicates the leading minor of order
///   info (and hence the matrix A itself) is not positive-definite,
///   and the factorization could not be completed.)
#define VSIP_IMPL_LAPACK_POTRF(T, FCN)					\
inline bool								\
potrf(char uplo, int n, T* a, int lda)					\
{									\
  int info;								\
  FCN(&uplo, &n, a, &lda, &info);					\
  if (info < 0)								\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::potrf -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
  return (info == 0);							\
}

VSIP_IMPL_LAPACK_POTRF(float,                spotrf_)
VSIP_IMPL_LAPACK_POTRF(double,               dpotrf_)
VSIP_IMPL_LAPACK_POTRF(std::complex<float>,  cpotrf_)
VSIP_IMPL_LAPACK_POTRF(std::complex<double>, zpotrf_)

#undef VSIP_IMPL_LAPACK_POTRF



/// POTRS - Solves a system of linear equations with a Cholesky-factored
/// symmetric (hermitian) postive-definite matrix.
///
/// :Returns:
///   true  if info == 0
///   false if info > 0,
///   (When info > 0, this indicates the leading minor of order
///   info (and hence the matrix A itself) is not positive-definite,
///   and the factorization could not be completed.)
#define VSIP_IMPL_LAPACK_POTRS(T, FCN)					\
inline void								\
potrs(char uplo, int n, int nhrs, T* a, int lda, T* b, int ldb)		\
{									\
  int info;								\
  FCN(&uplo, &n, &nhrs, a, &lda, b, &ldb, &info);			\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::potrs -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}

VSIP_IMPL_LAPACK_POTRS(float,                spotrs_)
VSIP_IMPL_LAPACK_POTRS(double,               dpotrs_)
VSIP_IMPL_LAPACK_POTRS(std::complex<float>,  cpotrs_)
VSIP_IMPL_LAPACK_POTRS(std::complex<double>, zpotrs_)

#undef VSIP_IMPL_LAPACK_POTRS



/// GETRF - compute LU factorization of a general matrix.
///
/// :Returns:
///   true  if info == 0
///   false if info > 0,
///   (When info > 0, this indicates the factorization has been
///   completed, but U is exactly singular.  Division by 0 will
///   occur if factor U is used to solve a system of linear eq.
#define VSIP_IMPL_LAPACK_GETRF(T, FCN)					\
inline bool								\
getrf(int m, int n, T* a, int lda, int* ipiv)				\
{									\
  int info;								\
  FCN(&m, &n, a, &lda, ipiv, &info);					\
  if (info < 0)								\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::getrf -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
  return (info == 0);							\
}

VSIP_IMPL_LAPACK_GETRF(float,                sgetrf_)
VSIP_IMPL_LAPACK_GETRF(double,               dgetrf_)
VSIP_IMPL_LAPACK_GETRF(std::complex<float>,  cgetrf_)
VSIP_IMPL_LAPACK_GETRF(std::complex<double>, zgetrf_)

#undef VSIP_IMPL_LAPACK_GETRF



/// GETRS - Solves a system of linear equations with a LU-factored
/// square matrix, with multiple right-hand sides.
#define VSIP_IMPL_LAPACK_GETRS(T, FCN)					\
inline void								\
getrs(char trans, int n, int nhrs, T* a, int lda, int* ipiv, T* b, int ldb) \
{									\
  int info;								\
  FCN(&trans, &n, &nhrs, a, &lda, ipiv, b, &ldb, &info);		\
  if (info != 0)							\
  {									\
    char msg[256];							\
    sprintf(msg, "lapack::getrs -- illegal arg (info=%d)", info);	\
    VSIP_IMPL_THROW(vsip::impl::unimplemented(msg));			\
  }									\
}

VSIP_IMPL_LAPACK_GETRS(float,                sgetrs_)
VSIP_IMPL_LAPACK_GETRS(double,               dgetrs_)
VSIP_IMPL_LAPACK_GETRS(std::complex<float>,  cgetrs_)
VSIP_IMPL_LAPACK_GETRS(std::complex<double>, zgetrs_)

#undef VSIP_IMPL_LAPACK_GETRS



} // namespace vsip::impl::lapack
} // namespace vsip::impl
} // namespace vsip

#undef USE_F2C_COMPLEX_RETURN
#undef FORTRAN_FLOAT_RETURN

#endif // VSIP_IMPL_LAPACK_HPP
