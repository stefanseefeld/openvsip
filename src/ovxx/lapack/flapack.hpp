//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_flapack_hpp_
#define ovxx_lapack_flapack_hpp_

#include <vsip/support.hpp>

namespace ovxx
{
namespace lapack
{

// LAPACK requires column-major layout
typedef vsip::col2_type required_order_type;

extern "C"
{
  void sgeqrf_(int*, int*, float*, int*, float*, float*, int*, int*);
  void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);
  void cgeqrf_(int*, int*, std::complex<float>*, int*, std::complex<float>*, std::complex<float>*, int*, int*);
  void zgeqrf_(int*, int*, std::complex<double>*, int*, std::complex<double>*, std::complex<double>*, int*, int*);

  void sgeqr2_(int*, int*, float*, int*, float*, float*, int*);
  void dgeqr2_(int*, int*, double*, int*, double*, double*, int*);
  void cgeqr2_(int*, int*, std::complex<float>*, int*, std::complex<float>*, std::complex<float>*, int*);
  void zgeqr2_(int*, int*, std::complex<double>*, int*, std::complex<double>*, std::complex<double>*, int*);

  void sormqr_(char*, char*, int*, int*, int*, float*, int*, float*, float*, int*, float*, int*, int*);
  void dormqr_(char*, char*, int*, int*, int*, double*, int*, double*, double*, int*, double*, int*, int*);
  void cunmqr_(char*, char*, int*, int*, int*, std::complex<float>*, int*, std::complex<float>*, std::complex<float>*, int*, std::complex<float>*, int*, int*);
  void zunmqr_(char*, char*, int*, int*, int*, std::complex<double>*, int*, std::complex<double>*, std::complex<double>*, int*, std::complex<double>*, int*, int*);

  void sgebrd_(int*, int*, float*, int*, float*, float*, float*, float*, float*, int*, int*);
  void dgebrd_(int*, int*, double*, int*, double*, double*, double*, double*, double*, int*, int*);
  void cgebrd_(int*, int*, std::complex<float>*, int*, float*, float*, std::complex<float>*, std::complex<float>*, std::complex<float>*, int*, int*);
  void zgebrd_(int*, int*, std::complex<double>*, int*, double*, double*, std::complex<double>*, std::complex<double>*, std::complex<double>*, int*, int*);

  void sorgbr_(char*, int*, int*, int*, float*, int*, float*, float*, int*, int*);
  void dorgbr_(char*, int*, int*, int*, double*, int*, double*, double*, int*, int*);
  void cungbr_(char*, int*, int*, int*, std::complex<float>*, int*, std::complex<float>*, std::complex<float>*, int*, int*);
  void zungbr_(char*, int*, int*, int*, std::complex<double>*, int*, std::complex<double>*, std::complex<double>*, int*, int*);

  void sbdsqr_(char*, int*, int*, int*, int*, float*, float*, float*, int*, float*, int*, float*, int*, float*, int*);
  void dbdsqr_(char*, int*, int*, int*, int*, double*, double*, double*, int*, double*, int*, double*, int*, double*, int*);
  void cbdsqr_(char*, int*, int*, int*, int*, float*, float*, std::complex<float>*, int*, std::complex<float>*, int*, std::complex<float>*, int*, std::complex<float>*, int*);
  void zbdsqr_(char*, int*, int*, int*, int*, double*, double*, std::complex<double>*, int*, std::complex<double>*, int*, std::complex<double>*, int*, std::complex<double>*, int*);

  void spotrf_(char*, int*, float*, int*, int*);
  void dpotrf_(char*, int*, double*, int*, int*);
  void cpotrf_(char*, int*, std::complex<float>*, int*, int*);
  void zpotrf_(char*, int*, std::complex<double>*, int*, int*);

  void spotrs_(char*, int*, int*, float*, int*, float*, int*, int*);
  void dpotrs_(char*, int*, int*, double*, int*, double*, int*, int*);
  void cpotrs_(char*, int*, int*, std::complex<float>*, int*, std::complex<float>*, int*, int*);
  void zpotrs_(char*, int*, int*, std::complex<double>*, int*, std::complex<double>*, int*, int*);

  void sgetrf_(int*, int*, float*, int*, int*, int*);
  void dgetrf_(int*, int*, double*, int*, int*, int*);
  void cgetrf_(int*, int*, std::complex<float>*, int*, int*, int*);
  void zgetrf_(int*, int*, std::complex<double>*, int*, int*, int*);

  void sgetrs_(char*, int*, int*, float*, int*, int*, float*, int*, int*);
  void dgetrs_(char*, int*, int*, double*, int*, int*, double*, int*, int*);
  void cgetrs_(char*, int*, int*, std::complex<float>*, int*, int*, std::complex<float>*, int*, int*);
  void zgetrs_(char*, int*, int*, std::complex<double>*, int*, int*, std::complex<double>*, int*, int*);
  int ilaenv_(int*, char const *, char const *, int*, int*, int*, int*);
}

inline int
ilaenv(int ispec, char const *name, char const *opts, int n1, int n2, int n3, int n4)
{
  return ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4);
}

#define OVXX_LAPACK_INVALID_ARG abort()

template <typename T> inline int geqrf_work(int m, int n);

#define OVXX_LAPACK_GEQRF(T, B)					\
inline void geqrf(int m, int n, T* a, int lda, T* tau, T* work,	\
		  int& lwork)					\
{								\
  int info;						      	\
  B##_(&m, &n, a, &lda, tau, work, &lwork, &info);		\
  if (info != 0)       						\
    abort();							\
}								\
template <>						       	\
inline int					       		\
geqrf_work<T>(int m, int n)       				\
{ return ilaenv(1, #B, "", m, n, -1, -1);}


OVXX_LAPACK_GEQRF(float,                sgeqrf)
OVXX_LAPACK_GEQRF(double,               dgeqrf)
OVXX_LAPACK_GEQRF(std::complex<float>,  cgeqrf)
OVXX_LAPACK_GEQRF(std::complex<double>, zgeqrf)

#undef OVXX_LAPACK_GEQRT

template <typename T> inline int geqr2_work(int m, int n);

#define OVXX_LAPACK_GEQR2(T, B)				        \
inline void geqr2(int m, int n, T* a, int lda, T* tau, T* work,	\
                  int& /*lwork*/)				\
{								\
  int info;							\
  B##_(&m, &n, a, &lda, tau, work, &info);			\
  if (info != 0)       						\
    OVXX_LAPACK_INVALID_ARG;					\
}								\
template <>						       	\
inline int					       		\
geqr2_work<T>(int m, int n)       				\
{ return ilaenv(1, #B, "", m, n, -1, -1);}

OVXX_LAPACK_GEQR2(float,                sgeqr2)
OVXX_LAPACK_GEQR2(double,               dgeqr2)
OVXX_LAPACK_GEQR2(std::complex<float>,  cgeqr2)
OVXX_LAPACK_GEQR2(std::complex<double>, zgeqr2)

#undef OVXX_LAPACK_GEQR2

template <typename T>
inline int
mqr_work(char side, char trans, int m, int n, int k);

#define OVXX_LAPACK_MQR(T, B)  					\
inline void mqr(char side, char trans,				\
		int m, int n, int k,				\
		T const *a, int lda,			       	\
		T *tau,				       		\
		T *c, int ldc,		       			\
		T *work, int& lwork)   				\
{			       					\
  int info;	       						\
  B##_(&side, &trans, &m, &n, &k, const_cast<T*>(a), &lda,	\
    tau, c, &ldc, work, &lwork, &info);				\
  if (info != 0)	       					\
    OVXX_LAPACK_INVALID_ARG;					\
}      								\
template <>    							\
inline int     							\
mqr_work<T>(char side, char trans, int m, int n, int k)   	\
{						       		\
  char arg[4]; arg[0] = side; arg[1] = trans; arg[2] = 0;      	\
  return ilaenv(1, #B, arg, m, n, k, -1);	       		\
}

OVXX_LAPACK_MQR(float,                sormqr)
OVXX_LAPACK_MQR(double,               dormqr)
OVXX_LAPACK_MQR(std::complex<float>,  cunmqr)
OVXX_LAPACK_MQR(std::complex<double>, zunmqr)

#undef OVXX_LAPACK_MQR

template <typename T> inline int gebrd_work(int m, int n);

#define OVXX_LAPACK_GEBRD(T, B)	       				\
inline void gebrd(int m, int n, T* a, int lda, 			\
		  scalar_of<T >::type* d,			\
		  scalar_of<T >::type* e,			\
		  T* tauq, T* taup, T* work, int& lwork)       	\
{						       		\
  int info;				       			\
  B##_(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);	\
  if (info != 0)				       		\
    OVXX_LAPACK_INVALID_ARG;					\
}								\
template <>    							\
inline int     							\
gebrd_work<T>(int m, int n)					\
{ return ilaenv(1, #B, "", m, n, -1, -1);}

OVXX_LAPACK_GEBRD(float,                sgebrd)
OVXX_LAPACK_GEBRD(double,               dgebrd)
OVXX_LAPACK_GEBRD(std::complex<float>,  cgebrd)
OVXX_LAPACK_GEBRD(std::complex<double>, zgebrd)

#undef OVXX_LAPACK_GEBRD

template <typename T>
inline int gbr_work(char vect, int m, int n, int k);

#define OVXX_LAPACK_GBR(T, B)			                \
inline void gbr(char vect,				       	\
		int m, int n, int k,		       		\
		T *a, int lda,		       			\
		T *tau,		       				\
		T *work, int& lwork)   				\
{			       					\
  int info;	       						\
  B##_(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);	\
  if (info != 0)				       		\
    OVXX_LAPACK_INVALID_ARG;					\
}								\
template <>					       		\
inline int				       			\
gbr_work<T>(char vect, int m, int n, int k)			\
{					       			\
  char arg[2]; arg[0] = vect; arg[1] = 0;      			\
  return ilaenv(1, #B, arg, m, n, k, -1);    			\
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
  int info;	       						\
  B##_(&uplo, &n, &ncvt, &nru, &ncc, d, e,			\
    vt, &ldvt, u, &ldu, c, &ldc, work, &info);		       	\
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
  int info;						       	\
  B##_(&uplo, &n, a, &lda, &info);				\
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
  int info;			       				\
  B##_(&uplo, &n, &nhrs, const_cast<T*>(a), &lda, b, &ldb, &info);\
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
  int info;		       					\
  B##_(&m, &n, a, &lda, ipiv, &info);				\
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
  int info;			       				\
  B##_(&trans, &n, &nhrs,					\
    const_cast<T*>(a), &lda, ipiv, b, &ldb, &info);    		\
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
