/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   Bindings for CUDA's BLAS functions.

#ifndef vsip_opt_cuda_blas_hpp_
#define vsip_opt_cuda_blas_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <complex>
#include <vsip/support.hpp>


#ifndef NDEBUG
#include <iostream>

// This macro should be used after any CUBLAS call to verify the
// call worked as expected.  For release builds, this check is
// not performed.
#define ASSERT_CUBLAS_OK()                                      \
{                                                               \
  cublasStatus status = cublasGetError();                       \
  if (status != 0)                                              \
    std::cerr << "CUBLAS problem encountered (error "           \
              << status << ")" << std::endl;                    \
  assert(status == 0);                                          \
}

#else
#define ASSERT_CUBLAS_OK()

#endif // NDEBUG

#if !defined(CUBLAS_H_)
extern "C"
{
typedef unsigned int cublasStatus;
cublasStatus cublasGetError();
} // extern "C"
#endif



namespace vsip
{
namespace impl
{
namespace cuda
{

#if !defined(CUBLAS_H_)

extern "C"
{
// Prototypes for CUBLAS functions called directly (see cublas.h)
// 
typedef int              I;
typedef char             T;
typedef float            S;
typedef double           D;
typedef cuComplex        C;
typedef cuDoubleComplex  Z;

S cublasSdot(I, const S*, I, const S*, I);
D cublasDdot(I, const D*, I, const D*, I);
C cublasCdotu(I, const C*, I, const C*, I);
C cublasCdotc(I, const C*, I, const C*, I);
Z cublasZdotu(I, const Z*, I, const Z*, I);
Z cublasZdotc(I, const Z*, I, const Z*, I);

void cublasStrsm(T, T, T, T, I, I, S, S const*, I, S*, I);
void cublasDtrsm(T, T, T, T, I, I, D, D const*, I, D*, I);
void cublasCtrsm(T, T, T, T, I, I, C, C const*, I, C*, I);
void cublasZtrsm(T, T, T, T, I, I, Z, Z const*, I, Z*, I);

void cublasSgemv(T, I, I, S, S const*, I, S const*, I, S, S*, I);
void cublasDgemv(T, I, I, D, D const*, I, D const*, I, D, D*, I);
void cublasCgemv(T, I, I, C, C const*, I, C const*, I, C, C*, I);
void cublasZgemv(T, I, I, Z, Z const*, I, Z const*, I, Z, Z*, I);

void cublasSgemm(T, T, I, I, I, S, S const*, I, S const*, I, S, S*, I);
void cublasDgemm(T, T, I, I, I, D, D const*, I, D const*, I, D, D*, I);
void cublasCgemm(T, T, I, I, I, C, C const*, I, C const*, I, C, C*, I);
void cublasZgemm(T, T, I, I, I, Z, Z const*, I, Z const*, I, Z, Z*, I);

void cublasSger(I, I, S, S const*, I, S const*, I, S*, I);
void cublasDger(I, I, D, D const*, I, D const*, I, D*, I);
void cublasCgerc(I, I, C, C const*, I, C const*, I, C*, I);
void cublasZgerc(I, I, Z, Z const*, I, Z const*, I, Z*, I);
void cublasCgeru(I, I, C, C const*, I, C const*, I, C*, I);
void cublasZgeru(I, I, Z, Z const*, I, Z const*, I, Z*, I);

} // extern "C"

#endif // !defined(CUBLAS_H_)


//
// C++ --> C interface functions
//

// cuda::dot()
#define CUBLAS_DOT(T, CUDA_T, VPPFCN, CUBLASFCN)                        \
inline T                                                                \
VPPFCN(int n, T const *x, int incx, T const *y, int incy)               \
{                                                                       \
  if (incx < 0) x += incx * (n-1);                                      \
  if (incy < 0) y += incy * (n-1);                                      \
  return CUBLASFCN(n, (CUDA_T const *)x, incx, (CUDA_T const *)y, incy);\
  ASSERT_CUBLAS_OK();                                                   \
}

#define CUBLAS_DOTC(T, CUDA_T, VPPFCN, CUBLASFCN)                           \
inline T                                                                    \
VPPFCN(int n, T const *x, int incx, T const *y, int incy)                   \
{                                                                           \
  if (incx < 0) x += incx * (n-1);                                          \
  if (incy < 0) y += incy * (n-1);                                          \
  CUDA_T r = CUBLASFCN(n, (CUDA_T const *)x, incx, (CUDA_T const *)y, incy);\
  ASSERT_CUBLAS_OK();                                                       \
  return T(r.x, r.y);                                                       \
}

// Note: CUDA functions return the C99 Complex type.  The way the
// return value is handled when converting back to the C++ type relies
// on a GNU extension and may not work with all compilers.
CUBLAS_DOT (float, float, dot,  cublasSdot)
CUBLAS_DOT (double, double, dot, cublasDdot)
CUBLAS_DOTC(std::complex<float>, cuComplex, dot,  cublasCdotu)
CUBLAS_DOTC(std::complex<float>, cuComplex, dotc, cublasCdotc)
CUBLAS_DOTC(std::complex<double>, cuDoubleComplex, dot,  cublasZdotu)
CUBLAS_DOTC(std::complex<double>, cuDoubleComplex, dotc, cublasZdotc)
#undef CUBLAS_DOT
#undef CUBLAS_DOTC



// cuda::trsm() (solver)
#define CUBLAS_TRSM(T, CUDA_T, FCN)                     \
inline void                                             \
trsm(char side, char uplo, char transa, char diag,      \
     int m, int n,                                      \
     T alpha,                                           \
     T const *a, int lda,				\
     T *b, int ldb)                                     \
{                                                       \
  CUDA_T& cu_alpha = (CUDA_T&) alpha;                   \
  FCN(side, uplo, transa, diag,                         \
      m, n,                                             \
      cu_alpha,                                         \
      (CUDA_T const*)a, lda,                            \
      (CUDA_T*)b, ldb);                                 \
  ASSERT_CUBLAS_OK();                                   \
}

CUBLAS_TRSM(float, float, cublasStrsm)
CUBLAS_TRSM(double, double, cublasDtrsm)
CUBLAS_TRSM(std::complex<float>,  cuComplex, cublasCtrsm)
CUBLAS_TRSM(std::complex<double>, cuDoubleComplex, cublasZtrsm)
#undef CUBLAS_TRSM



// cuda::gemv()
#define CUBLAS_GEMV(T, CUDA_T, FCN)             \
inline void                                     \
gemv(char transa,                               \
     int m, int n,                              \
     T alpha,                                   \
     T const *a, int lda,			\
     T const *x, int incx,			\
     T beta,                                    \
     T *y, int incy)                            \
{                                               \
  CUDA_T& cu_alpha = (CUDA_T&) alpha;           \
  CUDA_T& cu_beta = (CUDA_T&) beta;             \
  FCN(transa,                                   \
      m, n,                                     \
      cu_alpha,                                 \
      (CUDA_T const*)a, lda,                    \
      (CUDA_T const*)x, incx,                   \
      cu_beta,                                  \
      (CUDA_T*)y, incy);                        \
  ASSERT_CUBLAS_OK();                           \
}

CUBLAS_GEMV(float, float, cublasSgemv)
CUBLAS_GEMV(double, double, cublasDgemv)
CUBLAS_GEMV(std::complex<float>, cuComplex, cublasCgemv)
CUBLAS_GEMV(std::complex<double>, cuDoubleComplex, cublasZgemv)
#undef CUBLAS_GEMV



// cuda::gemm()
#define CUBLAS_GEMM(T, CUDA_T, FCN)             \
inline void                                     \
gemm(char transa, char transb,                  \
     int m, int n, int k,                       \
     T alpha,                                   \
     T const *a, int lda,			\
     T const *b, int ldb,			\
     T beta,                                    \
     T *c, int ldc)                             \
{                                               \
  CUDA_T& cu_alpha = (CUDA_T&) alpha;           \
  CUDA_T& cu_beta = (CUDA_T&) beta;             \
  FCN(transa, transb,                           \
      m, n, k,                                  \
      cu_alpha,                                 \
      (CUDA_T const*)a, lda,                    \
      (CUDA_T const*)b, ldb,                    \
      cu_beta,                                  \
      (CUDA_T*)c, ldc);                         \
  ASSERT_CUBLAS_OK();                           \
}

CUBLAS_GEMM(float, float, cublasSgemm)
CUBLAS_GEMM(double, double, cublasDgemm)
CUBLAS_GEMM(std::complex<float>, cuComplex, cublasCgemm)
CUBLAS_GEMM(std::complex<double>, cuDoubleComplex, cublasZgemm)
#undef CUBLAS_GEMM



// cuda:ger()
#define CUBLAS_GER(T, CUDA_T, VPPFCN, FCN)      \
inline void                                     \
VPPFCN(int m, int n,                            \
       T alpha,					\
       T const *x, int incx,			\
       T const *y, int incy,			\
       T *a, int lda)				\
{                                               \
  CUDA_T& cu_alpha = (CUDA_T&) alpha;           \
  FCN(m, n,                                     \
      cu_alpha,                                 \
      (CUDA_T const*)x, incx,                   \
      (CUDA_T const*)y, incy,                   \
      (CUDA_T*)a, lda);                         \
  ASSERT_CUBLAS_OK();                           \
}

CUBLAS_GER(float, float, ger, cublasSger)
CUBLAS_GER(double, double, ger, cublasDger)
CUBLAS_GER(std::complex<float>,  cuComplex,       gerc, cublasCgerc)
CUBLAS_GER(std::complex<double>, cuDoubleComplex, gerc, cublasZgerc)
CUBLAS_GER(std::complex<float>,  cuComplex,       geru, cublasCgeru)
CUBLAS_GER(std::complex<double>, cuDoubleComplex, geru, cublasZgeru)
#undef CUBLAS_GER


} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif
