/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   BLAS evaluators (for use in general dispatch).

#ifndef VSIP_OPT_LAPACK_MATVEC_HPP
#define VSIP_OPT_LAPACK_MATVEC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/complex.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/lapack/bindings.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// BLAS evaluator for vector-vector outer product
template <typename T1,
	  typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::outer, be::blas,
                 void(Block0&, T1, Block1 const&, Block2 const&)>
{
  static bool const ct_valid = 
    impl::blas::Blas_traits<T1>::valid &&
    is_same<T1, typename Block1::value_type>::value &&
    is_same<T1, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& /*r*/, T1 /*alpha*/,
		       Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;

    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    return (data_a.stride(0) == 1) && (data_b.stride(0) == 1);
  }

  static void exec(Block0& r, T1 alpha, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == r.size(2, 0));
    assert(b.size(1, 0) == r.size(2, 1));

    typedef typename get_block_layout<Block0>::order_type order0_type;

    dda::Data<Block0, dda::out> data_r(r);
    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    T1 const val = T1();
    impl::expr::Scalar<2, T1> scalar(val);
    impl::assign<2>(r, scalar);

    if (is_same<order0_type, row2_type>::value)
    {
      // Use identity:
      //   R = A B     <=>     trans(R) = trans(B) trans(A)

      impl::blas::ger( 
        b.size(1, 0), a.size(1, 0),     // int m, int n,
        alpha,                          // T alpha,
        data_b.ptr(), data_b.stride(0),  // T *x, int incx,
        data_a.ptr(), data_a.stride(0),  // T *y, int incy,
        data_r.ptr(), r.size(2, 1)      // T *a, int lda
      );
    }
    else if (is_same<order0_type, col2_type>::value)
    {
      impl::blas::ger( 
        a.size(1, 0), b.size(1, 0),     // int m, int n,
        alpha,                          // T alpha,
        data_a.ptr(), data_a.stride(0),  // T *x, int incx,
        data_b.ptr(), data_b.stride(0),  // T *y, int incy,
        data_r.ptr(), r.size(2, 0)      // T *a, int lda
      );
    }
    else
      assert(0);
  }
};

template <typename T1,
	  typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::outer, be::blas,
                 void(Block0&, complex<T1>, Block1 const&, Block2 const&)>
{
  static bool const ct_valid = 
    impl::blas::Blas_traits<complex<T1> >::valid &&
    is_same<complex<T1>, typename Block1::value_type>::value &&
    is_same<complex<T1>, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0& /*r*/, complex<T1> /*alpha*/, 
    Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;

    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    return (data_a.stride(0) == 1) && (data_b.stride(0) == 1);
  }

  static void exec(Block0& r, complex<T1> alpha, 
    Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == r.size(2, 0));
    assert(b.size(1, 0) == r.size(2, 1));

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::out> data_r(r);
    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    complex<T1> const val = complex<T1>();
    impl::expr::Scalar<2, complex<T1> > scalar(val);
    impl::assign<2>(r, scalar);

    if (is_same<order0_type, row2_type>::value)
    {
      // BLAS does not have a function that will conjugate the first 
      // vector and allow us to take advantage of the identity:
      //   R = A B*     <=>     trans(R) = trans(B*) trans(A)
      // This requires a manual conjugation after calling the library 
      // function.

      impl::blas::gerc( 
        b.size(1, 0), a.size(1, 0),     // int m, int n,
        complex<T1>(1),            // T alpha,
        data_b.ptr(), data_b.stride(0),  // T *x, int incx,
        data_a.ptr(), data_a.stride(0),  // T *y, int incy,
        data_r.ptr(), r.size(2, 1)      // T *a, int lda
      );

      for (index_type i = 0; i < r.size(2, 0); ++i )
	for (index_type j = 0; j < r.size(2, 1); ++j )
	  r.put(i, j, alpha * conj(r.get(i, j)));

    }
    else if (is_same<order0_type, col2_type>::value)
    {
      impl::blas::gerc( 
        a.size(1, 0), b.size(1, 0),     // int m, int n,
        alpha,                          // T alpha,
        data_a.ptr(), data_a.stride(0),  // T *x, int incx,
        data_b.ptr(), data_b.stride(0),  // T *y, int incy,
        data_r.ptr(), r.size(2, 0)      // T *a, int lda
      );
    }
    else
      assert(0);
  }
};


/// BLAS evaluator for vector-vector dot-product (non-conjugated).
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::blas,
                 T(Block0 const&, Block1 const&)>
{
  static bool const ct_valid = 
    impl::blas::Blas_traits<T>::valid &&
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value;

  static bool rt_valid(Block0 const&, Block1 const&) { return true; }

  static T exec(Block0 const& a, Block1 const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    dda::Data<Block0, dda::in> data_a(a);
    dda::Data<Block1, dda::in> data_b(b);

    T r = impl::blas::dot(a.size(1, 0),
			  data_a.ptr(), data_a.stride(0),
			  data_b.ptr(), data_b.stride(0));

    return r;
  }
};


/// BLAS evaluator for vector-vector dot-product (conjugated).
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::blas,
                 complex<T>(Block0 const&, 
                            expr::Unary<expr::op::Conj, Block1, true> const&)>
{
  static bool const ct_valid = 
    impl::blas::Blas_traits<complex<T> >::valid &&
    is_same<complex<T>, typename Block0::value_type>::value &&
    is_same<complex<T>, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value;

  static bool rt_valid(Block0 const&, 
		       expr::Unary<expr::op::Conj, Block1, true> const&)
  { return true; }

  static complex<T> exec(Block0 const& a, 
			 expr::Unary<expr::op::Conj, Block1, true> const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    dda::Data<Block0, dda::in> data_a(a);
    dda::Data<Block1, dda::in> data_b(b.arg());

    return impl::blas::dotc(a.size(1, 0),
			    data_b.ptr(), data_b.stride(0),
			    data_a.ptr(), data_a.stride(0));
    // Note:
    //   BLAS    cdotc(x, y)  => conj(x) * y, while 
    //   VSIPL++ cvjdot(x, y) => x * conj(y)
  }
};


/// BLAS evaluator for matrix-vector product
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::blas,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 1>::type>
{
  typedef typename Block0::value_type T;

  static bool const ct_valid = 
    impl::blas::Blas_traits<T>::valid &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0& /*r*/, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;

    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    // Note: gemm is used for row type and b is restricted to unit stride.
    // gemv is used for col type and can handle any stride for b.
    bool is_a_row = is_same<order1_type, row2_type>::value;
    return is_a_row ? ((data_a.stride(1) == 1) && (data_b.stride(0) == 1))
                    : (data_a.stride(0) == 1);
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(2, 1) == b.size(1, 0));

    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::out> data_r(r);
    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    if (is_same<order1_type, row2_type>::value)
    {
      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      char transa   = 'n';           // already transposed
      char transb   = 'n';

      impl::blas::gemm(
        transa, transb,
        1,                  // M
        a.size(2, 0),       // N
        a.size(2, 1),       // K
        1.0,                // alpha
        data_b.ptr(), data_b.stride(0), // vector, first dim is implicitly 1
        data_a.ptr(), a.size(2, 1),
        0.0,                // beta
        data_r.ptr(), data_r.stride(0)  // vector, first dim is implicitly 1
      );
    }
    else if (is_same<order1_type, col2_type>::value)
    {
      char transa   = 'n';

      impl::blas::gemv( 
        transa,                          // char trans,
        a.size(2, 0), a.size(2, 1),      // int m, int n,
        1.0,                             // T alpha,
        data_a.ptr(), data_a.stride(1),  // T *a, int lda,
        data_b.ptr(), data_b.stride(0),  // T *x, int incx,
        0.0,                             // T beta,
        data_r.ptr(), data_r.stride(0)   // T *y, int incy)
      );
    }
    else assert(0);
  }
};


/// BLAS evaluator for vector-matrix product
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::blas,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 1 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  static bool const ct_valid = 
    impl::blas::Blas_traits<T>::valid &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0& /*r*/, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    // Note: gemv is used for row type and can handle any stride for a.
    // gemm is used for col type and a is restricted to unit stride.
    // 
    bool is_b_row = is_same<order2_type, row2_type>::value;
    return is_b_row ? (data_b.stride(1) == 1) 
                    : ((data_b.stride(0) == 1) && (data_a.stride(0) == 1));
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == b.size(2, 0));

    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::out> data_r(r);
    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    if (is_same<order2_type, row2_type>::value)
    {
      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      char transa   = 'n';

      impl::blas::gemv( 
        transa,                          // char trans,
        b.size(2, 1), b.size(2, 0),      // int m, int n,
        1.0,                             // T alpha,
        data_b.ptr(), data_b.stride(0),   // T *a, int lda,
        data_a.ptr(), data_a.stride(0),   // T *x, int incx,
        0.0,                             // T beta,
        data_r.ptr(), data_r.stride(0)    // T *y, int incy)
      );
    }
    else if (is_same<order2_type, col2_type>::value)
    {
      char transa   = 'n';
      char transb   = 'n';

      impl::blas::gemm(
        transa, transb,
        1,                  // M
        b.size(2, 1),       // N
        b.size(2, 0),       // K
        1.0,                // alpha
        data_a.ptr(), data_a.stride(0),    // vector, first dim is implicitly 1
        data_b.ptr(), b.size(2, 0),
        0.0,                // beta
        data_r.ptr(), data_r.stride(0)     // vector, first dim is implicitly 1
      );
    }
    else assert(0);
  }
};


/// BLAS evaluator for matrix-matrix products.
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::blas,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  static bool const is_block0_interleaved =
    !impl::is_complex<typename Block0::value_type>::value ||
    get_block_layout<Block0>::storage_format == interleaved_complex;
  static bool const is_block1_interleaved =
    !impl::is_complex<typename Block1::value_type>::value ||
    get_block_layout<Block1>::storage_format == interleaved_complex;
  static bool const is_block2_interleaved =
    !impl::is_complex<typename Block2::value_type>::value ||
    get_block_layout<Block2>::storage_format == interleaved_complex;

  static bool const ct_valid = 
    impl::blas::Blas_traits<T>::valid &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that data is interleaved
    is_block0_interleaved && is_block1_interleaved && is_block2_interleaved &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& /*r*/, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    bool is_a_row = is_same<order1_type, row2_type>::value;
    bool is_b_row = is_same<order2_type, row2_type>::value;

    return (data_a.stride(is_a_row ? 1 : 0) == 1 &&
            data_b.stride(is_b_row ? 1 : 0) == 1 );
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::out> data_r(r);
    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    if (is_same<order0_type, row2_type>::value)
    {
      bool is_a_row = is_same<order1_type, row2_type>::value;
      char transa   = is_a_row ? 'n' : 't';
      int  lda      = is_a_row ? data_a.stride(0) : data_a.stride(1);

      bool is_b_row = is_same<order2_type, row2_type>::value;
      char transb   = is_b_row ? 'n' : 't';
      int  ldb      = is_b_row ? data_b.stride(0) : data_b.stride(1);

      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      impl::blas::gemm(transb, transa,
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

      impl::blas::gemm(transa, transb,
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


/// BLAS evaluator for generalized matrix-matrix products.
template <typename T1,
	  typename T2,
	  typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::gemp, be::blas,
                 void(Block0&, T1, Block1 const&, Block2 const&, T2)>
{
  typedef typename Block0::value_type T;

  static bool const ct_valid = 
    impl::blas::Blas_traits<T>::valid &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0&, T1, Block1 const& a, Block2 const& b, T2)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    bool is_a_row = is_same<order1_type, row2_type>::value;
    bool is_b_row = is_same<order2_type, row2_type>::value;

    return (data_a.stride(is_a_row ? 1 : 0) == 1 &&
            data_b.stride(is_b_row ? 1 : 0) == 1 );
  }

  static void exec(Block0& r, T1 alpha, Block1 const& a, 
		   Block2 const& b, T2 beta)
  {
    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::out> data_r(r);
    dda::Data<Block1, dda::in> data_a(a);
    dda::Data<Block2, dda::in> data_b(b);

    if (is_same<order0_type, row2_type>::value)
    {
      bool is_a_row = is_same<order1_type, row2_type>::value;
      char transa   = is_a_row ? 'n' : 't';
      int  lda      = is_a_row ? data_a.stride(0) : data_a.stride(1);

      bool is_b_row = is_same<order2_type, row2_type>::value;
      char transb   = is_b_row ? 'n' : 't';
      int  ldb      = is_b_row ? data_b.stride(0) : data_b.stride(1);

      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      impl::blas::gemm(transb, transa,
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

      impl::blas::gemm(transa, transb,
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

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_LAPACK_MATVEC_HPP
