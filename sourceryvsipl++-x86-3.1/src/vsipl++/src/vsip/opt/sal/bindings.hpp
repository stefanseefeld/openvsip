/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/bindings.hpp
    @author  Don McCoy
    @date    2005-10-04
    @brief   VSIPL++ Library: Wrappers and traits to bridge with 
               Mercury SAL.
*/

#ifndef VSIP_OPT_SAL_BINDINGS_HPP
#define VSIP_OPT_SAL_BINDINGS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <complex>
#include <sal.h>

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/opt/sal/elementwise.hpp>
#include <vsip/opt/sal/eval_elementwise.hpp>
#include <vsip/opt/sal/eval_threshold.hpp>
#include <vsip/opt/sal/eval_vcmp.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace sal
{

#define VSIP_IMPL_DOT(T, SAL_T, VPPFCN, SALFCN, STRIDE)	\
inline T					\
VPPFCN(length_type len,				\
       T const *A, stride_type A_stride,	\
       T const *B, stride_type B_stride)	\
{						\
  T c;						\
  SALFCN( (SAL_T *) A, STRIDE * A_stride,	\
          (SAL_T *) B, STRIDE * B_stride,	\
          (SAL_T *) &c, len, 0 );		\
  return c;					\
}				

#define VSIP_IMPL_DOT_SPLIT(T, SAL_T, VPPFCN, SALFCN, STRIDE)	\
inline std::complex<T>					\
VPPFCN(length_type len,				        \
       std::pair<T const*, T const*> A, stride_type A_stride,	\
       std::pair<T const*, T const*> B, stride_type B_stride)	\
{							\
  T real_val, imag_val;					\
  SAL_T c = { &real_val, &imag_val };			\
  SALFCN( (SAL_T *) &A, STRIDE * A_stride,		\
          (SAL_T *) &B, STRIDE * B_stride,		\
          (SAL_T *) &c, len, 0 );			\
           						\
  std::complex<T> r(*c.realp, *c.imagp);		\
  return r;						\
}

VSIP_IMPL_DOT(float, float,                         dot, dotprx, 1);
VSIP_IMPL_DOT(double, double,                       dot, dotprdx, 1);
VSIP_IMPL_DOT(std::complex<float>, COMPLEX,         dot, cdotprx, 2);
VSIP_IMPL_DOT(std::complex<double>, DOUBLE_COMPLEX, dot, cdotprdx, 2);
VSIP_IMPL_DOT_SPLIT(float, COMPLEX_SPLIT,           dot, zdotprx, 1);
VSIP_IMPL_DOT_SPLIT(double, DOUBLE_COMPLEX_SPLIT,   dot, zdotprdx, 1);

VSIP_IMPL_DOT(std::complex<float>, COMPLEX,         dotc, cidotprx, 2);
VSIP_IMPL_DOT(std::complex<double>, DOUBLE_COMPLEX, dotc, cidotprdx, 2);
VSIP_IMPL_DOT_SPLIT(float, COMPLEX_SPLIT,           dotc, zidotprx, 1);
VSIP_IMPL_DOT_SPLIT(double, DOUBLE_COMPLEX_SPLIT,   dotc, zidotprdx, 1);

#undef VSIP_IMPL_DOT
#undef VSIP_IMPL_DOT_SPLIT



// Functions that take col-stride parameters, where rows must be unit-stride.
// Also supports transpose and accumulate operations.

#define VSIP_IMPL_MAT_PROD(T, SAL_T, SALFCN) \
inline void                       \
mat_mul(int nr_c, int nc_r,       \
	int n,			  \
	T const *a, int a_tcols,  \
	T const *b, int b_tcols,  \
	T *c, int c_tcols,	  \
	int mflag)		  \
{                                 \
  SALFCN( (SAL_T *) a, a_tcols,   \
          (SAL_T *) b, b_tcols,   \
          (SAL_T *) c, c_tcols,   \
          nr_c, nc_r, n,          \
          mflag, 0 );             \
}

#define VSIP_IMPL_MAT_PROD_SPLIT( T, SAL_T, SALFCN ) \
inline void                                  \
mat_mul(int nr_c, int nc_r,                  \
	int n,				     \
	std::pair<T const*, T const*> a, int a_tcols,  \
	std::pair<T const*, T const*> b, int b_tcols,  \
	std::pair<T *, T *> c, int c_tcols,  \
	int mflag )			     \
{                                            \
  SALFCN((SAL_T *) &a, a_tcols,              \
	 (SAL_T *) &b, b_tcols,		     \
	 (SAL_T *) &c, c_tcols,		     \
	 nr_c, nc_r, n,			     \
	 mflag, 0 );			     \
}

VSIP_IMPL_MAT_PROD(float, float,                       mat_mulx);
VSIP_IMPL_MAT_PROD(double, double,                     mat_muldx);
VSIP_IMPL_MAT_PROD(complex<float>, COMPLEX,            cmat_mulx);
VSIP_IMPL_MAT_PROD(complex<double>, DOUBLE_COMPLEX,    cmat_muldx);
VSIP_IMPL_MAT_PROD_SPLIT(float, COMPLEX_SPLIT,         zmat_mulx);
VSIP_IMPL_MAT_PROD_SPLIT(double, DOUBLE_COMPLEX_SPLIT, zmat_muldx);

#undef VSIP_IMPL_MAT_PROD
#undef VSIP_IMPL_MAT_PROD_SPLIT

// Functions that support rows with variable stride. [Note: complex versions
// are deprecated according to Mercury SAL documentation, but are recommended
// at this time for good performance -- 2005-10-23.]

#define VSIP_IMPL_PROD( T, SAL_T, SALFCN, STRIDE_X ) \
inline void                            \
mmul(int m, int n, int p,              \
     T const *a, int as,	       \
     T const *b, int bs,	       \
     T *c, int cs )		       \
{                                      \
  SALFCN( (SAL_T *) a, STRIDE_X * as,  \
          (SAL_T *) b, STRIDE_X * bs,  \
          (SAL_T *) c, STRIDE_X * cs,  \
          m, n, p, 0 );                \
}

#define VSIP_IMPL_PROD_SPLIT( T, SAL_T, SALFCN ) \
inline void                            \
mmul(int m, int n, int p,	       \
     std::pair<T const*, T const*> a, int as,	 \
     std::pair<T const*, T const*> b, int bs,	 \
     std::pair<T *, T *> c, int cs )		 \
{                                      \
  SALFCN( (SAL_T *) &a, as,            \
          (SAL_T *) &b, bs,            \
          (SAL_T *) &c, cs,            \
          m, n, p, 0 );                \
}

VSIP_IMPL_PROD(float, float,                       mmulx, 1);
VSIP_IMPL_PROD(double, double,                     mmuldx, 1);
VSIP_IMPL_PROD(complex<float>, COMPLEX,            cmmulx, 2);
VSIP_IMPL_PROD(complex<double>, DOUBLE_COMPLEX,    cmmuldx, 2);
VSIP_IMPL_PROD_SPLIT(float, COMPLEX_SPLIT,         zmmulx);
VSIP_IMPL_PROD_SPLIT(double, DOUBLE_COMPLEX_SPLIT, zmmuldx);

#undef VSIP_IMPL_PROD
#undef VSIP_IMPL_PROD_SPLIT

template <typename T>
struct Sal_traits
{
  static bool const valid = false;
};

template <>
struct Sal_traits<float>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct Sal_traits<double>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct Sal_traits<std::complex<float> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

template <>
struct Sal_traits<std::complex<double> >
{
  static bool const valid = true;
  static char const trans = 'c';
};






// functions for vector multiply
inline void vmul(...) { VSIP_IMPL_THROW(unimplemented("vmul(...)")); }


// convolution functions

#define VSIP_IMPL_CONV( T, SAL_T, SALFCN, STRIDE_X )			\
inline void								\
conv(T const *filter, int f_as, int M,					\
     T const *input,  int i_as, int N,					\
     T *output, int o_as )						\
{									\
  SALFCN(								\
    (SAL_T *) &input[0],      /* input vector, length of A >= N+p-1 */	\
    i_as * STRIDE_X,          /* address stride for A               */	\
    (SAL_T *) &filter[M - 1], /* input filter                       */	\
    -1 * f_as * STRIDE_X,     /* address stride for B               */	\
    (SAL_T *) &output[0],     /* output vector                      */	\
    o_as * STRIDE_X,          /* address stride for C               */	\
    N,                        /* real output count                  */	\
    M,                        /* filter length (vector B)           */	\
    0                         /* ESAL flag                          */	\
  );                                   \
}

#define VSIP_IMPL_CONV_SPLIT(T, SAL_T, SALFCN, STRIDE_X )		\
inline void								\
conv(std::pair<T const*, T const*> filter, int f_as, int M,		\
     std::pair<T const*, T const*> input,  int i_as, int N,		\
     std::pair<T*, T*> output, int o_as)				\
{									\
  SAL_T filter_end = {(T*)filter.first + M-1, (T*)filter.second + M-1};	\
  SALFCN(								\
    (SAL_T *) &input,         /* input vector, length of A >= N+p-1 */	\
    i_as * STRIDE_X,          /* address stride for A               */	\
    (SAL_T *) &filter_end,    /* input filter                       */	\
    -1 * f_as * STRIDE_X,     /* address stride for B               */	\
    (SAL_T *) &output,        /* output vector                      */	\
    o_as * STRIDE_X,          /* address stride for C               */	\
    N,                        /* real output count                  */	\
    M,                        /* filter length (vector B)           */	\
    0                         /* ESAL flag                          */	\
    );									\
}

VSIP_IMPL_CONV(float,          float,         convx,  1);
VSIP_IMPL_CONV(complex<float>, COMPLEX,       cconvx, 2);

VSIP_IMPL_CONV_SPLIT(float, COMPLEX_SPLIT, zconvx, 1);

#undef VSIP_IMPL_CONV
#undef VSIP_IMPL_CONV_SPLIT

#define VSIP_IMPL_CONV_2D(T, SAL_T, SALFCN, STRIDE_X)			\
inline void								\
conv2d(T const *filter,							\
       int filter_rows, int filter_cols,				\
       int filter_row_stride ATTRIBUTE_UNUSED,				\
       T const *input,							\
       int input_rows ATTRIBUTE_UNUSED, int input_cols ATTRIBUTE_UNUSED, \
       int input_row_stride,						\
       T*  output,							\
       int output_rows, int output_cols,				\
       int output_row_stride,						\
       int row_dec,							\
       int col_dec)							\
{									\
  assert(filter_row_stride == filter_cols);				\
  assert(input_rows == (output_rows-1)*row_dec + filter_rows);		\
  assert(input_cols == (output_cols-1)*col_dec + filter_cols);		\
  SALFCN(								\
    (SAL_T*)&input[0],        /* input matrix                       */	\
    input_row_stride,         /* total columns in input matrix      */	\
    (SAL_T *)&filter[0],      /* kernel matrix                      */	\
    (SAL_T *)&output[0],      /* output matrix                      */	\
    output_row_stride,        /* total columns in output matrix     */	\
    row_dec,                  /* row decimation factor              */	\
    col_dec,                  /* column decimation factor           */	\
    filter_cols,              /* number of filter columns           */	\
    filter_rows,              /* number of filter rows              */	\
    output_cols,              /* number of output columns           */	\
    output_rows,              /* number of output rows              */	\
    0,                        /* reserved                           */	\
    0                         /* ESAL flag                          */	\
    );									\
}

VSIP_IMPL_CONV_2D(float, float, conv2dx, 1);

#undef VSIP_IMPL_CONV_2D

#define VSIP_IMPL_CONV_2D_3X3(T, SAL_T, SALFCN, STRIDE_X)		\
inline void								\
conv2d_3x3(T const *filter,						\
	   T const *input,						\
	   T *output,							\
	   int rows, int cols)						\
{									\
  SALFCN(								\
    (SAL_T*)&input[0],        /* input matrix                       */	\
    cols,                     /* number of in/out columns           */	\
    rows,                     /* number of in/out rows              */	\
    (SAL_T *)&filter[0],      /* 3x3 filter matrix                  */	\
    (SAL_T *)&output[0],      /* output matrix                      */	\
    0                         /* ESAL flag                          */	\
    );									\
}

VSIP_IMPL_CONV_2D_3X3(float, float, f3x3x, 1);

#undef VSIP_IMPL_CONV_2D_3X3

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

#endif
