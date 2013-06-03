//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_lapacke_hpp_
#define ovxx_lapack_lapacke_hpp_

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

#include <lapacke/lapacke.h>

namespace ovxx
{
namespace lapack
{

typedef any_type required_order_type;

#define OVXX_LAPACK_INVALID_ARG abort()

// Unfortunately, we can't use pre-allocated workspace with
// LAPACKE, because it requires the same knowledge (such as
// data layout) as the actual call.
// Some of this information is not known upfront.
// We also can't use the dimension ordering parameter, since
// it assumes / requires all matrix arguments to use the same,
// which may not be the case in general.

template <typename T> 
inline int geqrf_work(int, int) { return 0;}

#define OVXX_LAPACK_GEQRF(T, B)					\
inline void geqrf(int m, int n, T* a, int lda, T* tau, T* work,	\
		  int& lwork)					\
{								\
  int info =						      	\
    LAPACKE_##B(LAPACK_COL_MAJOR, m, n, a, lda, tau);		\
  if (info != 0)						\
    abort();							\
}

OVXX_LAPACK_GEQRF(float,                sgeqrf)
OVXX_LAPACK_GEQRF(double,               dgeqrf)
OVXX_LAPACK_GEQRF(std::complex<float>,  cgeqrf)
OVXX_LAPACK_GEQRF(std::complex<double>, zgeqrf)

#undef OVXX_LAPACK_GEQRT

template <typename T> 
inline int geqr2_work(int, int) { return 0;}

#define OVXX_LAPACK_GEQR2(T, B)				        \
inline void geqr2(int m, int n, T* a, int lda, T* tau, T* work,	\
                  int& /*lwork*/)				\
{								\
  int info =							\
    LAPACKE_##B(LAPACK_COL_MAJOR, m, n, a, lda, tau);		\
  if (info != 0)       						\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_GEQR2(float,                sgeqr2)
OVXX_LAPACK_GEQR2(double,               dgeqr2)
OVXX_LAPACK_GEQR2(std::complex<float>,  cgeqr2)
OVXX_LAPACK_GEQR2(std::complex<double>, zgeqr2)

#undef OVXX_LAPACK_GEQR2

template <typename T>
inline int
mqr_work(char, char, int, int, int) { return 0;}

#define OVXX_LAPACK_MQR(T, B)  					\
inline void mqr(char side, char trans,				\
		int m, int n, int k,				\
		T const *a, int lda,			       	\
		T *tau,				       		\
		T *c, int ldc,		       			\
		T *work, int& lwork)   				\
{			       					\
  int info =	       						\
    LAPACKE_##B(LAPACK_COL_MAJOR,				\
		side, trans, m, n, k, const_cast<T*>(a), lda,	\
		tau, c, ldc);					\
  if (info != 0)	       					\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_MQR(float,                sormqr)
OVXX_LAPACK_MQR(double,               dormqr)
OVXX_LAPACK_MQR(std::complex<float>,  cunmqr)
OVXX_LAPACK_MQR(std::complex<double>, zunmqr)

#undef OVXX_LAPACK_MQR

template <typename T> 
inline int gebrd_work(int, int) { return 0;}

#define OVXX_LAPACK_GEBRD(T, B)	       				\
inline void gebrd(int m, int n, T* a, int lda, 			\
		  scalar_of<T >::type* d,			\
		  scalar_of<T >::type* e,			\
		  T* tauq, T* taup, T* work, int& lwork)       	\
{						       		\
  int info =				       			\
    LAPACKE_##B(LAPACK_COL_MAJOR,				\
		m, n, a, lda, d, e, tauq, taup);		\
  if (info != 0)				       		\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_GEBRD(float,                sgebrd)
OVXX_LAPACK_GEBRD(double,               dgebrd)
OVXX_LAPACK_GEBRD(std::complex<float>,  cgebrd)
OVXX_LAPACK_GEBRD(std::complex<double>, zgebrd)

#undef OVXX_LAPACK_GEBRD

template <typename T>
inline int gbr_work(char, int, int, int) { return 0;}

#define OVXX_LAPACK_GBR(T, B)			                \
inline void gbr(char vect,				       	\
		int m, int n, int k,		       		\
		T *a, int lda,		       			\
		T *tau,		       				\
		T *work, int& lwork)   				\
{			       					\
  int info =	       						\
    LAPACKE_##B(LAPACK_COL_MAJOR, vect, m, n, k, a, lda, tau);	\
  if (info != 0)				       		\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_GBR(float,                sorgbr)
OVXX_LAPACK_GBR(double,               dorgbr)
OVXX_LAPACK_GBR(std::complex<float>,  cungbr)
OVXX_LAPACK_GBR(std::complex<double>, zungbr)

#undef OVXX_LAPACK_GBR

#define OVXX_LAPACK_BDSQR(T, B)					\
inline void bdsqr(char uplo,		       			\
		  int n, int ncvt, int nru, int ncc,		\
		  scalar_of<T >::type* d,			\
		  scalar_of<T >::type* e,			\
		  T *vt, int ldvt,			       	\
		  T *u, int ldu,		       		\
		  T *c, int ldc,	       			\
		  T *work)	       				\
{			       					\
  int info =	       						\
    LAPACKE_##B(LAPACK_COL_MAJOR, uplo, n, ncvt, nru, ncc, d, e,\
		vt, ldvt, u, ldu, c, ldc);		       	\
  if (info != 0)				       		\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_BDSQR(float,                sbdsqr)
OVXX_LAPACK_BDSQR(double,               dbdsqr)
OVXX_LAPACK_BDSQR(std::complex<float>,  cbdsqr)
OVXX_LAPACK_BDSQR(std::complex<double>, zbdsqr)

#undef OVXX_LAPACK_BDSQR

#define OVXX_LAPACK_POTRF(T, B)					\
inline bool    							\
potrf(char uplo, int n, T* a, int lda)				\
{      								\
  int info =						       	\
    LAPACKE_##B(LAPACK_COL_MAJOR, uplo, n, a, lda);		\
  if (info < 0)					       		\
    OVXX_LAPACK_INVALID_ARG;					\
  return info == 0;  						\
}

OVXX_LAPACK_POTRF(float,                spotrf)
OVXX_LAPACK_POTRF(double,               dpotrf)
OVXX_LAPACK_POTRF(std::complex<float>,  cpotrf)
OVXX_LAPACK_POTRF(std::complex<double>, zpotrf)

#undef OVXX_LAPACK_POTRF

#define OVXX_LAPACK_POTRS(T, B)      				\
inline void		       					\
potrs(char uplo, int n, int nhrs,			       	\
      T const *a, int lda, T* b, int ldb)	       		\
{					       			\
  int info =			       				\
    LAPACKE_##B(LAPACK_COL_MAJOR,				\
		uplo, n, nhrs, const_cast<T*>(a), lda, b, ldb);	\
  if (info != 0)						\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_POTRS(float,                spotrs)
OVXX_LAPACK_POTRS(double,               dpotrs)
OVXX_LAPACK_POTRS(std::complex<float>,  cpotrs)
OVXX_LAPACK_POTRS(std::complex<double>, zpotrs)

#undef OVXX_LAPACK_POTRS

#define OVXX_LAPACK_GETRF(T, B)					\
inline bool	       						\
getrf(int m, int n, T* a, int lda, int* ipiv)  			\
{				       				\
  int info =		       					\
    LAPACKE_##B(LAPACK_COL_MAJOR,				\
		m, n, a, lda, ipiv);				\
  if (info < 0)		       					\
    OVXX_LAPACK_INVALID_ARG;					\
  return info == 0;						\
}

OVXX_LAPACK_GETRF(float,                sgetrf)
OVXX_LAPACK_GETRF(double,               dgetrf)
OVXX_LAPACK_GETRF(std::complex<float>,  cgetrf)
OVXX_LAPACK_GETRF(std::complex<double>, zgetrf)

#undef OVXX_LAPACK_GETRF

#define OVXX_LAPACK_GETRS(T, B)					\
inline void	       						\
getrs(char trans, int n, int nhrs,			       	\
      T const *a, int lda, int* ipiv, T* b, int ldb)   		\
{					       			\
  int info =			       				\
    LAPACKE_##B(LAPACK_COL_MAJOR,				\
		trans, n, nhrs,					\
		const_cast<T*>(a), lda, ipiv, b, ldb);		\
  if (info != 0)			       			\
    OVXX_LAPACK_INVALID_ARG;					\
}

OVXX_LAPACK_GETRS(float,                sgetrs)
OVXX_LAPACK_GETRS(double,               dgetrs)
OVXX_LAPACK_GETRS(std::complex<float>,  cgetrs)
OVXX_LAPACK_GETRS(std::complex<double>, zgetrs)

#undef OVXX_LAPACK_GETRS

} // namespace ovxx::lapack
} // namespace ovxx

#endif
