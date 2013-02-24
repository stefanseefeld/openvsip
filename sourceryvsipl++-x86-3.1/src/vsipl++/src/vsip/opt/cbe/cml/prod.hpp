/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/cml/prod.hpp
    @author  Don McCoy
    @date    2008-05-07
    @brief   VSIPL++ Library: Bindings for CML matrix product routines.
*/

#ifndef VSIP_OPT_CBE_CML_PROD_HPP
#define VSIP_OPT_CBE_CML_PROD_HPP

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

#include <cml.h>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace cml
{


// This macro supports scalar and interleaved complex types

#define VSIP_IMPL_CML_VDOT(T, FCN, CML_FCN)			\
  inline void							\
FCN(T const *a, int lda, T const *b, int ldb, T *z, int n)	\
{								\
  typedef scalar_of<T>::type CML_T;				\
  CML_FCN(reinterpret_cast<CML_T const *>(a),			\
	  static_cast<ptrdiff_t>(lda),				\
	  reinterpret_cast<CML_T const *>(b),			\
	  static_cast<ptrdiff_t>(ldb),				\
	  reinterpret_cast<CML_T*>(z),				\
	  static_cast<size_t>(n) );				\
}

VSIP_IMPL_CML_VDOT(float,               dot,  cml_vdot_f)
VSIP_IMPL_CML_VDOT(std::complex<float>, dot,  cml_cvdot_f)
VSIP_IMPL_CML_VDOT(std::complex<float>, dotc, cml_cvdotj_f)
#undef VSIP_IMPL_CML_VDOT

// This version is for split complex only.

#define VSIP_IMPL_CML_ZVDOT(T, FCN, CML_FCN)	      \
inline void					      \
FCN(std::pair<T const *, T const *> a, int lda,	      \
    std::pair<T const *, T const *> b, int ldb,	      \
    std::complex<T>*  z,			      \
    int n)					      \
{						      \
  T z_real, z_imag;				      \
  CML_FCN(a.first, a.second,			      \
	  static_cast<ptrdiff_t>(lda),		      \
	  b.first, b.second,			      \
	  static_cast<ptrdiff_t>(ldb),		      \
	  &z_real, &z_imag,			      \
	  static_cast<size_t>(n) );		      \
  *z = std::complex<T>(z_real, z_imag);		      \
}

VSIP_IMPL_CML_ZVDOT(float, dot,  cml_zvdot_f)
VSIP_IMPL_CML_ZVDOT(float, dotc, cml_zvdotj_f)
#undef VSIP_IMPL_CML_ZVDOT


// This macro supports scalar and interleaved complex types

#define VSIP_IMPL_CML_VOUTER(T, FCN, CML_FCN)   \
inline void                                     \
FCN(T alpha,	        			\
    T const * a, int lda,                       \
    T const * b, int ldb,                       \
    T* z, int ldz,                              \
    int m,                                      \
    int n)                                      \
{						\
  typedef scalar_of<T>::type CML_T;		\
  CML_FCN(reinterpret_cast<CML_T const *>(a),	\
	  static_cast<ptrdiff_t>(lda),		\
	  reinterpret_cast<CML_T const *>(b),	\
	  static_cast<ptrdiff_t>(ldb),		\
	  reinterpret_cast<CML_T*>(z),		\
	  static_cast<ptrdiff_t>(ldz),		\
	  static_cast<size_t>(m),		\
	  static_cast<size_t>(n) );		\
  T* pz = z;					\
  for (int i = 0; i < m; ++i)			\
    for (int j = 0; j < n; ++j)			\
      pz[i*ldz + j] *= alpha;			\
}

VSIP_IMPL_CML_VOUTER(float,               outer,  cml_vouter_f)
VSIP_IMPL_CML_VOUTER(std::complex<float>, outer,  cml_cvouter_f)
#undef VSIP_IMPL_CML_VOUTER

// This version is for split complex only.

#define VSIP_IMPL_CML_ZVOUTER(T, FCN, CML_FCN)  \
inline void                                     \
FCN(std::complex<T>   alpha,                    \
    std::pair<T const *, T const *> a, int lda,	\
    std::pair<T const *, T const *> b, int ldb,	\
    std::pair<T*, T*> z, int ldz,               \
    int m,                                      \
    int n)                                      \
{						\
  CML_FCN(a.first, a.second,			\
	  static_cast<ptrdiff_t>(lda),		\
	  b.first, b.second,			\
	  static_cast<ptrdiff_t>(ldb),		\
	  z.first, z.second,			\
	  static_cast<ptrdiff_t>(ldz),		\
	  static_cast<size_t>(m),		\
	  static_cast<size_t>(n) );		\
  T* zr = z.first;				\
  T* zi = z.second;				\
  for (int i = 0; i < m; ++i)			\
    for (int j = 0; j < n; ++j)			\
    {                                           \
      T real = alpha.real() * zr[i*ldz + j]	\
	- alpha.imag() * zi[i*ldz + j];		\
      T imag = alpha.real() * zi[i*ldz + j]	\
	+ alpha.imag() * zr[i*ldz + j];		\
      zr[i*ldz + j] = real;			\
      zi[i*ldz + j] = imag;			\
    }						\
}

VSIP_IMPL_CML_ZVOUTER(float, outer, cml_zvouter_f)
#undef VSIP_IMPL_CML_ZVOUTER


// This macro supports scalar and interleaved complex types

#define VSIP_IMPL_CML_MVPROD(T, FCN, CML_FCN)   \
inline void                                     \
FCN(T const *a, int lda,			\
    T const *b, int ldb,			\
    T *z, int ldz,                              \
    int m, int n)                               \
{						\
  typedef scalar_of<T>::type CML_T;		\
  CML_FCN(reinterpret_cast<CML_T const *>(a),	\
	  static_cast<ptrdiff_t>(lda),		\
	  reinterpret_cast<CML_T const *>(b),	\
	  static_cast<ptrdiff_t>(ldb),		\
	  reinterpret_cast<CML_T*>(z),		\
	  static_cast<ptrdiff_t>(ldz),		\
	  static_cast<size_t>(m),		\
	  static_cast<size_t>(n) );		\
}

VSIP_IMPL_CML_MVPROD(float,               mvprod,  cml_mvprod1_f)
VSIP_IMPL_CML_MVPROD(std::complex<float>, mvprod,  cml_cmvprod1_f)
VSIP_IMPL_CML_MVPROD(float,               vmprod,  cml_vmprod1_f)
VSIP_IMPL_CML_MVPROD(std::complex<float>, vmprod,  cml_cvmprod1_f)
#undef VSIP_IMPL_CML_MVPROD

// This version is for split complex only.

#define VSIP_IMPL_CML_ZMVPROD(T, FCN, CML_FCN)  \
inline void                                     \
FCN(std::pair<T const *, T const *> a, int lda,	\
    std::pair<T const *, T const *> b, int ldb,	\
    std::pair<T*, T*> z, int ldz,               \
    int m, int n)                               \
{						\
  typedef scalar_of<T>::type CML_T;		\
  CML_FCN(a.first, a.second,			\
	  static_cast<ptrdiff_t>(lda),		\
	  b.first, b.second,			\
	  static_cast<ptrdiff_t>(ldb),		\
	  z.first, z.second,			\
	  static_cast<ptrdiff_t>(ldz),		\
	  static_cast<size_t>(m),		\
	  static_cast<size_t>(n) );		\
}

VSIP_IMPL_CML_ZMVPROD(float, mvprod, cml_zmvprod1_f)
VSIP_IMPL_CML_ZMVPROD(float, vmprod, cml_zvmprod1_f)
#undef VSIP_IMPL_CML_ZMVPROD



// This macro supports scalar and interleaved complex types

#define VSIP_IMPL_CML_MPROD(T, FCN, CML_FCN)    \
inline void                                     \
FCN(T const *a, int lda,			\
    T const *b, int ldb,			\
    T *z, int ldz,                              \
    int m, int n, int p)                        \
{						\
  typedef scalar_of<T>::type CML_T;		\
  CML_FCN(reinterpret_cast<CML_T const *>(a),	\
	  static_cast<ptrdiff_t>(lda),		\
	  reinterpret_cast<CML_T const *>(b),	\
	  static_cast<ptrdiff_t>(ldb),		\
	  reinterpret_cast<CML_T*>(z),		\
	  static_cast<ptrdiff_t>(ldz),		\
	  static_cast<size_t>(m),		\
	  static_cast<size_t>(n),		\
	  static_cast<size_t>(p) );		\
}

VSIP_IMPL_CML_MPROD(float,               mprod,  cml_mprod1_f)
VSIP_IMPL_CML_MPROD(float,               mprodt, cml_mprodt1_f)
VSIP_IMPL_CML_MPROD(std::complex<float>, mprod,  cml_cmprod1_f)
VSIP_IMPL_CML_MPROD(std::complex<float>, mprodt, cml_cmprodt1_f)
VSIP_IMPL_CML_MPROD(std::complex<float>, mprodj, cml_cmprodj1_f)
VSIP_IMPL_CML_MPROD(std::complex<float>, mprodh, cml_cmprodh1_f)
#undef VSIP_IMPL_CML_MPROD

// This version is for split complex only.

#define VSIP_IMPL_CML_ZMPROD(T, FCN, CML_FCN)   \
inline void                                     \
FCN(std::pair<T const *, T const *> a, int lda,	\
    std::pair<T const *, T const *> b, int ldb,	\
    std::pair<T*, T*> z, int ldz,               \
    int m, int n, int p)                        \
{						\
  CML_FCN(a.first, a.second,			\
	  static_cast<ptrdiff_t>(lda),		\
	  b.first, b.second,			\
	  static_cast<ptrdiff_t>(ldb),		\
	  z.first, z.second,			\
	  static_cast<ptrdiff_t>(ldz),		\
	  static_cast<size_t>(m),		\
	  static_cast<size_t>(n),		\
	  static_cast<size_t>(p) );		\
}

VSIP_IMPL_CML_ZMPROD(float, mprod,  cml_zmprod1_f)
VSIP_IMPL_CML_ZMPROD(float, mprodt, cml_zmprodt1_f)
VSIP_IMPL_CML_ZMPROD(float, mprodj, cml_zmprodj1_f)
VSIP_IMPL_CML_ZMPROD(float, mprodh, cml_zmprodh1_f)
#undef VSIP_IMPL_CML_ZMPROD


} // namespace vsip::impl::cml
} // namespace vsip::impl
} // namespace vsip

#endif
