//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_blas_hpp_
#define ovxx_lapack_blas_hpp_

#include <vsip/dda.hpp>
#include <ovxx/view/fns_elementwise.hpp>

namespace ovxx
{
namespace blas
{
extern "C"
{
float sdot_(int*, float*, int*, float*, int*);
double ddot_(int*, double*, int*, double*, int*);
float _Complex cdotu_(int*, std::complex<float>*, int*, std::complex<float>*, int*);
double _Complex zdotu_(int*, std::complex<double>*, int*, std::complex<double>*, int*);
float _Complex cdotc_(int*, std::complex<float>*, int*, std::complex<float>*, int*);
double _Complex zdotc_(int*, std::complex<double>*, int*, std::complex<double>*, int*);
void strsm_(char*, char*, char*, char*, int*, int*, float*, float*, int*, float*, int*);
void dtrsm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);
void ctrsm_(char*, char*, char*, char*, int*, int*, std::complex<float>*, std::complex<float>*, int*, std::complex<float>*, int*);
void ztrsm_(char*, char*, char*, char*, int*, int*, std::complex<double>*, std::complex<double>*, int*, std::complex<double>*, int*);

void sgemv_(char*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);
void dgemv_(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
void cgemv_(char*, int*, int*, std::complex<float>*, std::complex<float>*, int*, std::complex<float>*, int*, std::complex<float>*, std::complex<float>*, int*);
void zgemv_(char*, int*, int*, std::complex<double>*, std::complex<double>*, int*, std::complex<double>*, int*, std::complex<double>*, std::complex<double>*, int*);

void sgemm_(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
void cgemm_(char*, char*, int*, int*, int*, std::complex<float>*, std::complex<float>*, int*, std::complex<float>*, int*, std::complex<float>*, std::complex<float>*, int*);
void zgemm_(char*, char*, int*, int*, int*, std::complex<double>*, std::complex<double>*, int*, std::complex<double>*, int*, std::complex<double>*, std::complex<double>*, int*);

void sger_(int*, int*, float*, float*, int*, float*, int*, float*, int*);
void dger_( int*, int*, double*, double*, int*, double*, int*, double*, int*);
void cgerc_(int*, int*, std::complex<float>*, std::complex<float>*, int*, std::complex<float>*, int*, std::complex<float>*, int*);
void zgerc_(int*, int*, std::complex<double>*, std::complex<double>*, int*, std::complex<double>*, int*, std::complex<double>*, int*);
void cgeru_(int*, int*, std::complex<float>*, std::complex<float>*, int*, std::complex<float>*, int*, std::complex<float>*, int*);
void zgeru_(int*, int*, std::complex<double>*, std::complex<double>*, int*, std::complex<double>*, int*, std::complex<double>*, int*);
}

#define OVXX_BLAS_DOT(T, B)			       \
inline T					       \
dot(int n, T const *x, int incx, T const *y, int incy) \
{						       \
  if (incx < 0) x += incx * (n-1);		       \
  if (incy < 0) y += incy * (n-1);		       \
  return B(&n,					       \
	   const_cast<T*>(x), &incx,		       \
	   const_cast<T*>(y), &incy);		       \
}

OVXX_BLAS_DOT(float, sdot_)
OVXX_BLAS_DOT(double, ddot_)

#undef OVXX_BLAS_DOT

#define OVXX_BLAS_CDOT(T, F, B)			\
inline complex<T>			        \
F(int n,					\
  complex<T> const *x, int incx,		\
  complex<T> const *y, int incy)		\
{						\
  if (incx < 0) x += incx * (n-1);		\
  if (incy < 0) y += incy * (n-1);		\
  float __complex__ retn =			\
    B(&n,					\
      const_cast<complex<T>*>(x), &incx,	\
      const_cast<complex<T>*>(y), &incy); 	\
  return complex<T>(__real__ retn,		\
		    __imag__ retn);		\
}

OVXX_BLAS_CDOT(float, dot, cdotu_)
OVXX_BLAS_CDOT(double, dot, zdotu_)
OVXX_BLAS_CDOT(float, dotc, cdotc_)
OVXX_BLAS_CDOT(double, dotc, zdotc_)

#undef OVXX_BLAS_CDOT

#define OVXX_BLAS_TRSM(T, B)   			\
inline void					\
trsm(char side, char uplo,char transa,char diag,\
     int m, int n,				\
     T alpha,					\
     T const *a, int lda,			\
     T *b, int ldb)				\
{						\
  B(&side, &uplo, &transa, &diag,		\
    &m, &n,		     			\
    &alpha,  					\
    const_cast<T *>(a), &lda,			\
    b, &ldb);					\
}

OVXX_BLAS_TRSM(float,                strsm_)
OVXX_BLAS_TRSM(double,               dtrsm_)
OVXX_BLAS_TRSM(complex<float>,  ctrsm_)
OVXX_BLAS_TRSM(complex<double>, ztrsm_)

#undef OVXX_BLAS_TRSM

#define OVXX_BLAS_GEMV(T, B)			\
inline void					\
gemv(char transa,				\
     int m, int n, T alpha,			\
     T const *a, int lda,			\
     T const *x, int incx,			\
     T beta, T *y, int incy)			\
{						\
  B(&transa,					\
    &m, &n,					\
    &alpha,					\
    const_cast<T*>(a), &lda,			\
    const_cast<T*>(x), &incx,			\
    &beta,					\
    y, &incy);					\
}

OVXX_BLAS_GEMV(float,                sgemv_)
OVXX_BLAS_GEMV(double,               dgemv_)
OVXX_BLAS_GEMV(complex<float>,  cgemv_)
OVXX_BLAS_GEMV(complex<double>, zgemv_)

#undef OVXX_BLAS_GEMV

#define OVXX_BLAS_GEMM(T, B)			\
inline void					\
gemm(char transa, char transb,			\
     int m, int n, int k,			\
     T alpha,					\
     T const *a, int lda,			\
     T const *b, int ldb,			\
     T beta,					\
     T *c, int ldc)				\
{						\
  B(&transa, &transb,				\
    &m, &n, &k,					\
    &alpha,					\
    const_cast<T*>(a), &lda,			\
    const_cast<T*>(b), &ldb,			\
    &beta,					\
    c, &ldc);					\
}

OVXX_BLAS_GEMM(float,                sgemm_)
OVXX_BLAS_GEMM(double,               dgemm_)
OVXX_BLAS_GEMM(complex<float>,  cgemm_)
OVXX_BLAS_GEMM(complex<double>, zgemm_)

#undef OVXX_BLAS_GEMM

#define OVXX_BLAS_GER(T, F, B)			\
inline void					\
F(int m, int n,           		        \
  T alpha,					\
  T const *x, int incx,				\
  T const *y, int incy,				\
  T *a, int lda)				\
{						\
  B(&m, &n,					\
      &alpha,					\
      const_cast<T*>(x), &incx,			\
      const_cast<T*>(y), &incy,			\
      a, &lda);					\
}

OVXX_BLAS_GER(float,                ger, sger_)
OVXX_BLAS_GER(double,               ger, dger_)
OVXX_BLAS_GER(complex<float>,  gerc, cgerc_)
OVXX_BLAS_GER(complex<double>, gerc, zgerc_)
OVXX_BLAS_GER(complex<float>,  geru, cgeru_)
OVXX_BLAS_GER(complex<double>, geru, zgeru_)

#undef OVXX_BLAS_GER

template <typename T>
struct traits
{
  static bool const valid = false;
};

template <>
struct traits<float>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct traits<double>
{
  static bool const valid = true;
  static char const trans = 't';
};

template <>
struct traits<complex<float> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

template <>
struct traits<complex<double> >
{
  static bool const valid = true;
  static char const trans = 'c';
};

} // namespace ovxx::blas

namespace dispatcher
{

/// BLAS evaluator for vector-vector outer product
template <typename B0, typename T, typename B1, typename B2>
struct Evaluator<op::outer, be::blas,
                 void(B0 &, T, B1 const &, B2 const &)>
{
  static bool const ct_valid = 
    blas::traits<T>::valid &&
    is_same<T, typename B1::value_type>::value &&
    is_same<T, typename B2::value_type>::value &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    dda::Data<B2, dda::in>::ct_cost == 0;

  static bool rt_valid(B0 &, T, B1 const &a, B2 const &b)
  {
    typedef typename get_block_layout<B1>::order_type order1_type;

    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);
    return (data_a.stride(0) == 1) && (data_b.stride(0) == 1);
  }

  static void exec(B0 &r, T alpha, B1 const &a, B2 const &b)
  {
    OVXX_PRECONDITION(a.size(1, 0) == r.size(2, 0));
    OVXX_PRECONDITION(b.size(1, 0) == r.size(2, 1));

    typedef typename get_block_layout<B0>::order_type order0_type;

    dda::Data<B0, dda::out> data_r(r);
    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    expr::Scalar<2, T> scalar(T(0));
    assign<2>(r, scalar);

    if (is_same<order0_type, row2_type>::value)
    {
      blas::ger(b.size(1, 0), a.size(1, 0),    // int m, int n,
		alpha,                         // T alpha,
		data_b.ptr(), data_b.stride(0),// T *x, int incx,
		data_a.ptr(), data_a.stride(0),// T *y, int incy,
		data_r.ptr(), r.size(2, 1));   // T *a, int lda
    }
    else if (is_same<order0_type, col2_type>::value)
    {
      blas::ger(a.size(1, 0), b.size(1, 0),    // int m, int n,
		alpha,                         // T alpha,
		data_a.ptr(), data_a.stride(0),// T *x, int incx,
		data_b.ptr(), data_b.stride(0),// T *y, int incy,
		data_r.ptr(), r.size(2, 0));   // T *a, int lda
    }
    else
      assert(0);
  }
};

template <typename B0, typename T, typename B1, typename B2>
struct Evaluator<op::outer, be::blas,
                 void(B0&, complex<T>, B1 const&, B2 const&)>
{
  static bool const ct_valid = 
    blas::traits<complex<T> >::valid &&
    is_same<complex<T>, typename B1::value_type>::value &&
    is_same<complex<T>, typename B2::value_type>::value &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    dda::Data<B2, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value &&
    !is_split_block<B2>::value;

  static bool rt_valid(B0 &, complex<T>, B1 const &a, B2 const &b)
  {
    typedef typename get_block_layout<B1>::order_type order1_type;

    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    return (data_a.stride(0) == 1) && (data_b.stride(0) == 1);
  }

  static void exec(B0 &r, complex<T> alpha, B1 const &a, B2 const &b)
  {
    OVXX_PRECONDITION(a.size(1, 0) == r.size(2, 0));
    OVXX_PRECONDITION(b.size(1, 0) == r.size(2, 1));

    typedef typename get_block_layout<B0>::order_type order0_type;
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B0, dda::out> data_r(r);
    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    expr::Scalar<2, complex<T> > scalar(complex<T>(0));
    assign<2>(r, scalar);

    if (is_same<order0_type, row2_type>::value)
    {
      blas::gerc(b.size(1, 0), a.size(1, 0),    // int m, int n,
		 complex<T>(1),                 // T alpha,
		 data_b.ptr(), data_b.stride(0),// T *x, int incx,
		 data_a.ptr(), data_a.stride(0),// T *y, int incy,
		 data_r.ptr(), r.size(2, 1));   // T *a, int lda

      // FIXME: use element-wise conjugate
      for (index_type i = 0; i < r.size(2, 0); ++i)
	for (index_type j = 0; j < r.size(2, 1); ++j)
	  r.put(i, j, alpha * conj(r.get(i, j)));

    }
    else if (is_same<order0_type, col2_type>::value)
    {
      blas::gerc(a.size(1, 0), b.size(1, 0),    // int m, int n,
		 alpha,                         // T alpha,
		 data_a.ptr(), data_a.stride(0),// T *x, int incx,
		 data_b.ptr(), data_b.stride(0),// T *y, int incy,
		 data_r.ptr(), r.size(2, 0));   // T *a, int lda
    }
    else
      assert(0);
  }
};

template <typename T, typename B0, typename B1>
struct Evaluator<op::dot, be::blas, T(B0 const&, B1 const&)>
{
  static bool const ct_valid = 
    blas::traits<T>::valid &&
    is_same<T, typename B0::value_type>::value &&
    is_same<T, typename B1::value_type>::value &&
    dda::Data<B0, dda::in>::ct_cost == 0 &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value;

  static bool rt_valid(B0 const&, B1 const&) { return true;}

  static T exec(B0 const &a, B1 const &b)
  {
    OVXX_PRECONDITION(a.size(1, 0) == b.size(1, 0));

    dda::Data<B0, dda::in> data_a(a);
    dda::Data<B1, dda::in> data_b(b);

    T r = blas::dot(a.size(1, 0),
		    data_a.ptr(), data_a.stride(0),
		    data_b.ptr(), data_b.stride(0));

    return r;
  }
};

template <typename T, typename B0, typename B1>
struct Evaluator<op::dot, be::blas,
                 complex<T>(B0 const&, 
                            expr::Unary<expr::op::Conj, B1, true> const&)>
{
  static bool const ct_valid = 
    blas::traits<complex<T> >::valid &&
    is_same<complex<T>, typename B0::value_type>::value &&
    is_same<complex<T>, typename B1::value_type>::value &&
    dda::Data<B0, dda::in>::ct_cost == 0 &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value;

  static bool rt_valid(B0 const&, 
		       expr::Unary<expr::op::Conj, B1, true> const&)
  { return true;}

  static complex<T> exec(B0 const &a, 
			 expr::Unary<expr::op::Conj, B1, true> const &b)
  {
    OVXX_PRECONDITION(a.size(1, 0) == b.size(1, 0));

    dda::Data<B0, dda::in> data_a(a);
    dda::Data<B1, dda::in> data_b(b.arg());

    return blas::dotc(a.size(1, 0),
		      data_b.ptr(), data_b.stride(0),
		      data_a.ptr(), data_a.stride(0));
  }
};

template <typename B0, typename B1, typename B2>
struct Evaluator<op::prod, be::blas,
                 void(B0 &, B1 const &, B2 const &),
		 typename enable_if<B1::dim == 2 && B2::dim == 1>::type>
{
  typedef typename B0::value_type T;

  static bool const ct_valid = 
    blas::traits<T>::valid &&
    is_same<T, typename B1::value_type>::value &&
    is_same<T, typename B2::value_type>::value &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    dda::Data<B2, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value &&
    !is_split_block<B2>::value;

  static bool rt_valid(B0 &, B1 const &a, B2 const &b)
  {
    typedef typename get_block_layout<B1>::order_type order1_type;

    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    // Note: gemm is used for row type and b is restricted to unit stride.
    // gemv is used for col type and can handle any stride for b.
    bool is_a_row = is_same<order1_type, row2_type>::value;
    return is_a_row ? ((data_a.stride(1) == 1) && (data_b.stride(0) == 1))
                    : (data_a.stride(0) == 1);
  }

  static void exec(B0 &r, B1 const &a, B2 const &b)
  {
    OVXX_PRECONDITION(a.size(2, 1) == b.size(1, 0));

    typedef typename get_block_layout<B0>::order_type order0_type;
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B0, dda::out> data_r(r);
    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    if (is_same<order1_type, row2_type>::value)
    {
      blas::gemm('n', 'n',           // no trans
		 1,                  // M
		 a.size(2, 0),       // N
		 a.size(2, 1),       // K
		 1.0,                // alpha
		 data_b.ptr(), data_b.stride(0),
		 data_a.ptr(), a.size(2, 1),
		 0.0,                // beta
		 data_r.ptr(), data_r.stride(0));
    }
    else if (is_same<order1_type, col2_type>::value)
    {
      blas::gemv('n',                            // no trans,
		 a.size(2, 0), a.size(2, 1),     // int m, int n,
		 1.0,                            // T alpha,
		 data_a.ptr(), data_a.stride(1), // T *a, int lda,
		 data_b.ptr(), data_b.stride(0), // T *x, int incx,
		 0.0,                            // T beta,
		 data_r.ptr(), data_r.stride(0));// T *y, int incy)
    }
    else assert(0);
  }
};

template <typename B0, typename B1, typename B2>
struct Evaluator<op::prod, be::blas,
                 void(B0 &, B1 const &, B2 const &),
		 typename enable_if<B1::dim == 1 && B2::dim == 2>::type>
{
  typedef typename B0::value_type T;

  static bool const ct_valid = 
    blas::traits<T>::valid &&
    is_same<T, typename B1::value_type>::value &&
    is_same<T, typename B2::value_type>::value &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    dda::Data<B2, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value &&
    !is_split_block<B2>::value;

  static bool rt_valid(B0 &, B1 const &a, B2 const &b)
  {
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    // Note: gemv is used for row type and can handle any stride for a.
    // gemm is used for col type and a is restricted to unit stride.
    // 
    bool is_b_row = is_same<order2_type, row2_type>::value;
    return is_b_row ? (data_b.stride(1) == 1) 
                    : ((data_b.stride(0) == 1) && (data_a.stride(0) == 1));
  }

  static void exec(B0 &r, B1 const &a, B2 const &b)
  {
    OVXX_PRECONDITION(a.size(1, 0) == b.size(2, 0));

    typedef typename get_block_layout<B0>::order_type order0_type;
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B0, dda::out> data_r(r);
    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    if (is_same<order2_type, row2_type>::value)
    {
      blas::gemv('n',                             // no trans,
		 b.size(2, 1), b.size(2, 0),      // int m, int n,
		 1.0,                             // T alpha,
		 data_b.ptr(), data_b.stride(0),  // T *a, int lda,
		 data_a.ptr(), data_a.stride(0),  // T *x, int incx,
		 0.0,                             // T beta,
		 data_r.ptr(), data_r.stride(0)); // T *y, int incy)
    }
    else if (is_same<order2_type, col2_type>::value)
    {
      blas::gemm('n', 'n',           // no trans
		 1,                  // M
		 b.size(2, 1),       // N
		 b.size(2, 0),       // K
		 1.0,                // alpha
		 data_a.ptr(), data_a.stride(0),
		 data_b.ptr(), b.size(2, 0),
		 0.0,                // beta
		 data_r.ptr(), data_r.stride(0));
    }
    else assert(0);
  }
};

template <typename B0, typename B1, typename B2>
struct Evaluator<op::prod, be::blas,
                 void(B0 &, B1 const &, B2 const &),
		 typename enable_if<B1::dim == 2 && B2::dim == 2>::type>
{
  typedef typename B0::value_type T;

  static bool const is_block0_array = get_block_layout<B0>::storage_format == array;
  static bool const is_block1_array = get_block_layout<B1>::storage_format == array;
  static bool const is_block2_array = get_block_layout<B2>::storage_format == array;

  static bool const ct_valid = 
    blas::traits<T>::valid &&
    is_same<T, typename B1::value_type>::value &&
    is_same<T, typename B2::value_type>::value &&
    is_block0_array && is_block1_array && is_block2_array &&
    dda::Data<B0, dda::out>::ct_cost == 0 &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    dda::Data<B2, dda::in>::ct_cost == 0;

  static bool rt_valid(B0 &, B1 const &a, B2 const &b)
  {
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    bool is_a_row = is_same<order1_type, row2_type>::value;
    bool is_b_row = is_same<order2_type, row2_type>::value;

    return (data_a.stride(is_a_row ? 1 : 0) == 1 &&
            data_b.stride(is_b_row ? 1 : 0) == 1 );
  }

  static void exec(B0 &r, B1 const &a, B2 const &b)
  {
    typedef typename get_block_layout<B0>::order_type order0_type;
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B0, dda::out> data_r(r);
    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    if (is_same<order0_type, row2_type>::value)
    {
      bool is_a_row = is_same<order1_type, row2_type>::value;
      char transa   = is_a_row ? 'n' : 't';
      int  lda      = is_a_row ? data_a.stride(0) : data_a.stride(1);

      bool is_b_row = is_same<order2_type, row2_type>::value;
      char transb   = is_b_row ? 'n' : 't';
      int  ldb      = is_b_row ? data_b.stride(0) : data_b.stride(1);

      blas::gemm(transb, transa,
                 b.size(2, 1),  // N
                 a.size(2, 0),  // M
                 a.size(2, 1),  // K
                 1.0,           // alpha
                 data_b.ptr(), ldb,
                 data_a.ptr(), lda,
                 0.0,           // beta
                 data_r.ptr(), data_r.stride(0));
    }
    else if (is_same<order0_type, col2_type>::value)
    {
      bool is_a_col = is_same<order1_type, col2_type>::value;
      char transa   = is_a_col ? 'n' : 't';
      int  lda      = is_a_col ? data_a.stride(1) : data_a.stride(0);

      bool is_b_col = is_same<order2_type, col2_type>::value;
      char transb   = is_b_col ? 'n' : 't';
      int  ldb      = is_b_col ? data_b.stride(1) : data_b.stride(0);

      blas::gemm(transa, transb,
                 a.size(2, 0),  // M
                 b.size(2, 1),  // N
                 a.size(2, 1),  // K
                 1.0,           // alpha
                 data_a.ptr(), lda,
                 data_b.ptr(), ldb,
                 0.0,           // beta
                 data_r.ptr(), data_r.stride(1));
    }
    else assert(0);
  }
};

template <typename B0, typename T1, typename B1, typename B2, typename T2>
struct Evaluator<op::gemp, be::blas,
                 void(B0 &, T1, B1 const &, B2 const &, T2)>
{
  typedef typename B0::value_type T;

  static bool const ct_valid = 
    blas::traits<T>::valid &&
    is_same<T, typename B1::value_type>::value &&
    is_same<T, typename B2::value_type>::value &&
    dda::Data<B0, dda::out>::ct_cost == 0 &&
    dda::Data<B1, dda::in>::ct_cost == 0 &&
    dda::Data<B2, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value &&
    !is_split_block<B2>::value;

  static bool rt_valid(B0 &, T1, B1 const &a, B2 const &b, T2)
  {
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    bool is_a_row = is_same<order1_type, row2_type>::value;
    bool is_b_row = is_same<order2_type, row2_type>::value;

    return (data_a.stride(is_a_row ? 1 : 0) == 1 &&
            data_b.stride(is_b_row ? 1 : 0) == 1 );
  }

  static void exec(B0 &r, T1 alpha, B1 const &a, B2 const &b, T2 beta)
  {
    typedef typename get_block_layout<B0>::order_type order0_type;
    typedef typename get_block_layout<B1>::order_type order1_type;
    typedef typename get_block_layout<B2>::order_type order2_type;

    dda::Data<B0, dda::out> data_r(r);
    dda::Data<B1, dda::in> data_a(a);
    dda::Data<B2, dda::in> data_b(b);

    if (is_same<order0_type, row2_type>::value)
    {
      bool is_a_row = is_same<order1_type, row2_type>::value;
      char transa   = is_a_row ? 'n' : 't';
      int  lda      = is_a_row ? data_a.stride(0) : data_a.stride(1);

      bool is_b_row = is_same<order2_type, row2_type>::value;
      char transb   = is_b_row ? 'n' : 't';
      int  ldb      = is_b_row ? data_b.stride(0) : data_b.stride(1);

      blas::gemm(transb, transa,
                 b.size(2, 1),  // N
                 a.size(2, 0),  // M
                 a.size(2, 1),  // K
                 alpha,         // alpha
                 data_b.ptr(), ldb,
                 data_a.ptr(), lda,
                 beta,          // beta
                 data_r.ptr(), data_r.stride(0));
    }
    else if (is_same<order0_type, col2_type>::value)
    {
      bool is_a_col = is_same<order1_type, col2_type>::value;
      char transa   = is_a_col ? 'n' : 't';
      int  lda      = is_a_col ? data_a.stride(1) : data_a.stride(0);

      bool is_b_col = is_same<order2_type, col2_type>::value;
      char transb   = is_b_col ? 'n' : 't';
      int  ldb      = is_b_col ? data_b.stride(1) : data_b.stride(0);

      blas::gemm(transa, transb,
                 a.size(2, 0),  // M
                 b.size(2, 1),  // N
                 a.size(2, 1),  // K
                 alpha,         // alpha
                 data_a.ptr(), lda,
                 data_b.ptr(), ldb,
                 beta,          // beta
                 data_r.ptr(), data_r.stride(1));
    }
    else assert(0);
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx


#endif
