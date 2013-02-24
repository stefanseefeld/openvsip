/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/matvec.hpp
    @author  Don McCoy
    @date    2009-02-05
    @brief   VSIPL++ Library: CUDA-based BLAS evaluators 
*/

#ifndef VSIP_OPT_CUDA_MATVEC_HPP
#define VSIP_OPT_CUDA_MATVEC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/cuda/blas.hpp>
#include <vsip/opt/cuda/dda.hpp>

// CUDA prodj accomplished by wrapping CUDA conj kernel along with GEMM.
namespace vsip
{
namespace impl
{
namespace cuda
{
extern void conj(std::complex<float> const* input,
                 std::complex<float>*       output,
                 length_type                length);

template <typename T, typename O>
inline void prodj(T          *output,
                       T          *temp_storage,
                       T const    *input_a,
                       T const    *input_b,
                       length_type a_num_rows,
                       length_type a_num_cols,
                       length_type b_num_cols,
                       stride_type in_a_leading_stride,
                       stride_type in_b_leading_stride,
                       stride_type out_leading_stride,
                       char        operation_a,
                       char        operation_b);

template<>
inline void prodj<float, row2_type>(
  float       *output,
  float       *temp,
  float const *input_a,
  float const *input_b,
  length_type  a_nrows,
  length_type  a_ncols,
  length_type  b_ncols,
  stride_type  a_stride,
  stride_type  b_stride,
  stride_type  r_stride,
  char         transa,
  char         transb)
{
  gemm(transb, transa,
       b_ncols,
       a_nrows,
       a_ncols,
       1.0,
       input_b, b_stride,
       input_a, a_stride,
       0.0,
       output, r_stride);
}

template<>
inline void prodj<float, col2_type>(
  float       *output,
  float       *temp,
  float const *input_a,
  float const *input_b,
  length_type  a_nrows,
  length_type  a_ncols,
  length_type  b_ncols,
  stride_type  a_stride,
  stride_type  b_stride,
  stride_type  r_stride,
  char         transa,
  char         transb)
{
  gemm(transa, transb,
       a_nrows,
       b_ncols,
       a_ncols,
       1.0,
       input_a, a_stride,
       input_b, b_stride,
       0.0,
       output, r_stride);
}

template<>
inline void prodj<std::complex<float>, col2_type>(
  std::complex<float>       *output,
  std::complex<float>       *temp,
  std::complex<float> const *input_a,
  std::complex<float> const *input_b,
  length_type                a_nrows,
  length_type                a_ncols,
  length_type                b_ncols,
  stride_type                a_stride,
  stride_type                b_stride,
  stride_type                r_stride,
  char                       transa,
  char                       transb)
{
  conj(input_b, temp, a_ncols * b_ncols);

  gemm(transa, transb,
       a_nrows,
       b_ncols,
       a_ncols,
       1.0,
       input_a, a_stride,
       temp, b_stride,
       0.0,
       output, r_stride);
}

template<>
inline void prodj<std::complex<float>, row2_type>(
  std::complex<float>       *output,
  std::complex<float>       *temp,
  std::complex<float> const *input_a,
  std::complex<float> const *input_b,
  length_type                a_nrows,
  length_type                a_ncols,
  length_type                b_ncols,
  stride_type                a_stride,
  stride_type                b_stride,
  stride_type                r_stride,
  char                       transa,
  char                       transb)
{
  conj(input_b, temp, a_ncols * b_ncols);

  gemm(transa, transb,
       b_ncols,
       a_nrows,
       a_ncols,
       1.0,
       temp, b_stride,
       input_a, a_stride,
       0.0,
       output, r_stride);
}

}// vsip::impl::cuda
}// vsip::impl
}// vsip

namespace vsip_csl
{
namespace dispatcher
{

template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prodj, be::cuda,
                 void(Block0&, Block1 const&, Block2 const&)>
{
  typedef typename Block0::value_type T;
  typedef typename get_block_layout<Block2>::type layout2_type;

  static bool const ct_valid = 
    impl::cuda::Traits<T>::valid &&
    impl::is_same<T, typename Block1::value_type>::value &&
    impl::is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value &&
    // check that the block is dense.
    layout2_type::packing == dense;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::in> dda_r(r);
    dda::Data<Block1, dda::in> dda_a(a);
    dda::Data<Block2, dda::in> dda_b(b);

    // Blocks may not alias one another.
    bool is_not_aliased = !(impl::is_same_ptr(dda_r.ptr(), dda_b.ptr()) ||
                            impl::is_same_ptr(dda_r.ptr(), dda_a.ptr()));

    bool is_a_row = impl::is_same<order1_type, row2_type>::value;
    bool is_b_row = impl::is_same<order2_type, row2_type>::value;

    return (is_not_aliased && 
	    dda_a.stride(is_a_row ? 1 : 0) == 1 &&
	    dda_b.stride(is_b_row ? 1 : 0) == 1);
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    typedef typename get_block_layout<Block2>::type layout2_type;
    dimension_type const dim = layout2_type::dim;

    impl::cuda::dda::Data<Block0, dda::out> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);

    //  Allocate temporary space on the device to place the conjugate of the
    //   B matrix if required.
    impl::cuda::Device_storage<RT, layout2_type>
             storage(impl::Applied_layout<layout2_type>(impl::extent<dim>(b)));

    bool is_a_row = impl::is_same<order1_type, row2_type>::value;
    char transa   = impl::is_same<order1_type, order0_type>::value ? 'n' : 't';
    int  lda      = is_a_row ? dev_a.stride(0) : dev_a.stride(1);

    bool is_b_row = impl::is_same<order2_type, row2_type>::value;
    char transb   = impl::is_same<order2_type, order0_type>::value ? 'n' : 't';
    int  ldb      = is_b_row ? dev_b.stride(0) : dev_b.stride(1);

    bool is_r_row = impl::is_same<order0_type, row2_type>::value;
    int  ldr      = is_r_row ? dev_r.stride(0) : dev_r.stride(1);

    impl::cuda::prodj<RT, order0_type>(dev_r.ptr(), storage.ptr(),
                                       dev_a.ptr(), dev_b.ptr(),
                                       a.size(2, 0), a.size(2, 1),
                                       b.size(2, 1), lda, ldb, ldr,
                                       transa, transb);
  }
};

/// CUDA evaluator for vector-vector outer product
template <typename T1,
	  typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::outer, be::cuda,
                 void(Block0&, T1, Block1 const&, Block2 const&)>
{
  static bool const ct_valid =
    impl::cuda::Traits<T1>::valid &&
    impl::is_same<T1, typename Block1::value_type>::value &&
    impl::is_same<T1, typename Block2::value_type>::value &&
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

    typedef typename Block0::value_type value0_type;
    typedef typename get_block_layout<Block0>::order_type order0_type;

    impl::cuda::dda::Data<Block0, dda::out> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);


    if (impl::is_same<order0_type, row2_type>::value)
    {
      impl::cuda::assign_scalar(value0_type(), dev_r.ptr(), 
        dev_r.stride(0), r.size(2,0), r.size(2,1));

      // Use identity:
      //   R = A B     <=>     trans(R) = trans(B) trans(A)
      impl::cuda::ger( 
        b.size(1, 0), a.size(1, 0),     // int m, int n,
        alpha,                          // T alpha,
        dev_b.ptr(), dev_b.stride(0),  // T *x, int incx,
        dev_a.ptr(), dev_a.stride(0),  // T *y, int incy,
        dev_r.ptr(), r.size(2, 1)      // T *a, int lda
      );
    }
    else if (impl::is_same<order0_type, col2_type>::value)
    {
      impl::cuda::assign_scalar(value0_type(), dev_r.ptr(), 
        dev_r.stride(1), r.size(2,1), r.size(2,0));

      impl::cuda::ger( 
        a.size(1, 0), b.size(1, 0),     // int m, int n,
        alpha,                          // T alpha,
        dev_a.ptr(), dev_a.stride(0),  // T *x, int incx,
        dev_b.ptr(), dev_b.stride(0),  // T *y, int incy,
        dev_r.ptr(), r.size(2, 0)      // T *a, int lda
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
struct Evaluator<op::outer, be::cuda,
                 void(Block0&, complex<T1>, Block1 const&, Block2 const&)>
{
  static bool const ct_valid = 
    impl::cuda::Traits<complex<T1> >::valid &&
    impl::is_same<complex<T1>, typename Block1::value_type>::value &&
    impl::is_same<complex<T1>, typename Block2::value_type>::value &&
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

    typedef typename Block0::value_type value0_type;
    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    impl::cuda::dda::Data<Block0, dda::out> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);

    if (impl::is_same<order0_type, row2_type>::value)
    {
      impl::cuda::assign_scalar(value0_type(), dev_r.ptr(), 
        dev_r.stride(0), r.size(2,0), r.size(2,1));

      // BLAS does not have a function that will conjugate the first 
      // vector and allow us to take advantage of the identity:
      //   R = A B*     <=>     trans(R) = trans(B*) trans(A)
      // This requires a manual conjugation after calling the library 
      // function.
      impl::cuda::gerc( 
        b.size(1, 0), a.size(1, 0),     // int m, int n,
        complex<T1>(1),                 // T alpha,
        dev_b.ptr(), dev_b.stride(0),  // T *x, int incx,
        dev_a.ptr(), dev_a.stride(0),  // T *y, int incy,
        dev_r.ptr(), r.size(2, 1)      // T *a, int lda
      );

      for (index_type i = 0; i < r.size(2, 0); ++i )
	for (index_type j = 0; j < r.size(2, 1); ++j )
	  r.put(i, j, alpha * conj(r.get(i, j)));
    }
    else if (impl::is_same<order0_type, col2_type>::value)
    {
      impl::cuda::assign_scalar(value0_type(), dev_r.ptr(), 
        dev_r.stride(1), r.size(2,1), r.size(2,0));

      impl::cuda::gerc( 
        a.size(1, 0), b.size(1, 0),     // int m, int n,
        alpha,                          // T alpha,
        dev_a.ptr(), dev_a.stride(0),  // T *x, int incx,
        dev_b.ptr(), dev_b.stride(0),  // T *y, int incy,
        dev_r.ptr(), r.size(2, 0)      // T *a, int lda
      );
    }
    else
      assert(0);
  }
};


/// CUDA evaluator for vector-vector dot-product (non-conjugated).
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::cuda,
                 T(Block0 const&, Block1 const&)>
{
  static bool const ct_valid =
    impl::cuda::Traits<T>::valid &&
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value;

  static bool rt_valid(Block0 const& a, Block1 const& b) 
  { 
    dda::Data<Block0, dda::in> data_a(a);
    dda::Data<Block1, dda::in> data_b(b);
    
    // check that data is unit stride
    return ((data_a.stride(0) == 1) && (data_b.stride(0) == 1));
  }

  static T exec(Block0 const& a, Block1 const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    impl::cuda::dda::Data<Block0, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block1, dda::in> dev_b(b);

    T r = impl::cuda::dot(a.size(1, 0),
			  dev_a.ptr(), a.stride(1, 0),
			  dev_b.ptr(), b.stride(1, 0));

    return r;
  }
};


/// CUDA evaluator for vector-vector dot-product (conjugated).
template <typename Block0, typename Block1>
struct Evaluator<op::dot, be::cuda,
  typename Block0::value_type(Block0 const&, expr::Unary<expr::op::Conj, Block1, true> const&)>
{
  typedef typename Block0::value_type value_type;

  static bool const ct_valid = 
    impl::cuda::Traits<value_type>::valid &&
    is_same<value_type, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value;

  static bool rt_valid(Block0 const &a,
		       expr::Unary<expr::op::Conj, Block1, true> const &b)
  {
    dda::Data<Block0, dda::in> data_a(a);
    dda::Data<Block1, dda::in> data_b(b.arg());
    
    // check that data is unit stride
    return ((data_a.stride(0) == 1) && (data_b.stride(0) == 1));
  }

  static value_type exec(Block0 const &a, 
			 expr::Unary<expr::op::Conj, Block1, true> const &b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    impl::cuda::dda::Data<Block0, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block1, dda::in> dev_b(b.arg());

    value_type r = impl::cuda::dotc(a.size(1, 0),
				    dev_b.ptr(), b.arg().stride(1, 0), 
				    dev_a.ptr(), a.stride(1, 0));
    // Note:
    //   BLAS    cdotc(x, y)  => conj(x) * y, while 
    //   VSIPL++ cvjdot(x, y) => x * conj(y)

    return r;
  }
};



/// CUDA evaluator for matrix-vector product
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cuda,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 1>::type>
{
  typedef typename Block0::value_type T;

  static bool const ct_valid = 
    impl::cuda::Traits<T>::valid &&
    impl::is_same<T, typename Block1::value_type>::value &&
    impl::is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;

    dda::Data<Block0, dda::in> dda_r(r);
    dda::Data<Block1, dda::in> dda_a(a);
    dda::Data<Block2, dda::in> dda_b(b);

    // Blocks may not alias one another.
    bool is_not_aliased = !impl::is_same_ptr(dda_r.ptr(), dda_b.ptr());

    // Note: gemm is used for row type and b is restricted to unit stride.
    // gemv is used for col type and can handle any stride for b.
    bool is_a_row = impl::is_same<order1_type, row2_type>::value;
    return is_not_aliased &&
           is_a_row ? ((dda_a.stride(1) == 1) && (dda_b.stride(0) == 1))
                    : (dda_a.stride(0) == 1);
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(2, 1) == b.size(1, 0));

    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    impl::cuda::dda::Data<Block0, dda::out> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);

    if (impl::is_same<order1_type, row2_type>::value)
    {
      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      char transa   = 'n';           // already transposed
      char transb   = 'n';

      impl::cuda::gemm(transa, transb,
		       1,                  // M
		       a.size(2, 0),       // N
		       a.size(2, 1),       // K
		       1.0,                // alpha
		       dev_b.ptr(), 1,     // vector, first dim is implicitly 1
		       dev_a.ptr(), a.size(2, 1),
		       0.0,                // beta
		       dev_r.ptr(), 1);    // vector, first dim is implicitly 1
    }
    else if (impl::is_same<order1_type, col2_type>::value)
    {
      char transa   = 'n';

      impl::cuda::gemv(transa,                          // char trans,
		       a.size(2, 0), a.size(2, 1),      // int m, int n,
		       1.0,                             // T alpha,
		       dev_a.ptr(), dev_a.stride(1),    // T *a, int lda,
		       dev_b.ptr(), dev_b.stride(0),    // T *x, int incx,
		       0.0,                             // T beta,
		       dev_r.ptr(), dev_r.stride(0));   // T *y, int incy)
    }
    else assert(0);
  }
};


/// CUDA evaluator for vector-matrix product
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cuda,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 1 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  static bool const ct_valid = 
    impl::cuda::Traits<T>::valid &&
    impl::is_same<T, typename Block1::value_type>::value &&
    impl::is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::in> dda_r(r);
    dda::Data<Block1, dda::in> dda_a(a);
    dda::Data<Block2, dda::in> dda_b(b);

    // Blocks may not alias one another.
    bool is_not_aliased = !impl::is_same_ptr(dda_r.ptr(), dda_a.ptr());

    // Note: gemv is used for row type and can handle any stride for a.
    // gemm is used for col type and a is restricted to unit stride.
    // 
    bool is_b_row = impl::is_same<order2_type, row2_type>::value;
    return is_not_aliased &&
           is_b_row ? (dda_b.stride(1) == 1) 
                    : ((dda_b.stride(0) == 1) && (dda_a.stride(0) == 1));
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == b.size(2, 0));

    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    impl::cuda::dda::Data<Block0, dda::out> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);

    if (impl::is_same<order2_type, row2_type>::value)
    {
      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      char transa   = 'n';

      impl::cuda::gemv(transa,                          // char trans,
		       b.size(2, 1), b.size(2, 0),      // int m, int n,
		       1.0,                             // T alpha,
		       dev_b.ptr(), dev_b.stride(0),    // T *a, int lda,
		       dev_a.ptr(), dev_a.stride(0),    // T *x, int incx,
		       0.0,                             // T beta,
		       dev_r.ptr(), dev_r.stride(0));   // T *y, int incy)
    }
    else if (impl::is_same<order2_type, col2_type>::value)
    {
      char transa   = 'n';
      char transb   = 'n';

      impl::cuda::gemm(transa, transb,
		       1,                  // M
		       b.size(2, 1),       // N
		       b.size(2, 0),       // K
		       1.0,                // alpha
		       dev_a.ptr(), 1,     // vector, first dim is implicitly 1
		       dev_b.ptr(), b.size(2, 0),
		       0.0,                // beta
		       dev_r.ptr(), 1);    // vector, first dim is implicitly 1
    }
    else assert(0);
  }
};


/// CUDA evaluator for matrix-matrix products.
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cuda,
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
    impl::cuda::Traits<T>::valid &&
    impl::is_same<T, typename Block1::value_type>::value &&
    impl::is_same<T, typename Block2::value_type>::value &&
    // check that data is interleaved
    is_block0_interleaved && is_block1_interleaved && is_block2_interleaved &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::in> dda_r(r);
    dda::Data<Block1, dda::in> dda_a(a);
    dda::Data<Block2, dda::in> dda_b(b);

    // Blocks may not alias one another.
    bool is_not_aliased = 
      !impl::is_same_ptr(dda_r.ptr(), dda_b.ptr()) &&
      !impl::is_same_ptr(dda_r.ptr(), dda_a.ptr());

    bool is_a_row = impl::is_same<order1_type, row2_type>::value;
    bool is_b_row = impl::is_same<order2_type, row2_type>::value;

    return (is_not_aliased && 
	    dda_a.stride(is_a_row ? 1 : 0) == 1 &&
	    dda_b.stride(is_b_row ? 1 : 0) == 1);
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    impl::cuda::dda::Data<Block0, dda::out> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);

    if (impl::is_same<order0_type, row2_type>::value)
    {
      bool is_a_row = impl::is_same<order1_type, row2_type>::value;
      char transa   = is_a_row ? 'n' : 't';
      int  lda      = is_a_row ? dev_a.stride(0) : dev_a.stride(1);

      bool is_b_row = impl::is_same<order2_type, row2_type>::value;
      char transb   = is_b_row ? 'n' : 't';
      int  ldb      = is_b_row ? dev_b.stride(0) : dev_b.stride(1);

      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      impl::cuda::gemm(transb, transa,
		       b.size(2, 1),  // N
		       a.size(2, 0),  // M
		       a.size(2, 1),  // K
		       1.0,           // alpha
		       dev_b.ptr(), ldb,
		       dev_a.ptr(), lda,
		       0.0,           // beta
		       dev_r.ptr(), dev_r.stride(0));
    }
    else if (impl::is_same<order0_type, col2_type>::value)
    {
      bool is_a_col = impl::is_same<order1_type, col2_type>::value;
      char transa   = is_a_col ? 'n' : 't';
      int  lda      = is_a_col ? dev_a.stride(1) : dev_a.stride(0);

      bool is_b_col = impl::is_same<order2_type, col2_type>::value;
      char transb   = is_b_col ? 'n' : 't';
      int  ldb      = is_b_col ? dev_b.stride(1) : dev_b.stride(0);

      impl::cuda::gemm(transa, transb,
		       a.size(2, 0),  // M
		       b.size(2, 1),  // N
		       a.size(2, 1),  // K
		       1.0,           // alpha
		       dev_a.ptr(), lda,
		       dev_b.ptr(), ldb,
		       0.0,           // beta
		       dev_r.ptr(), dev_r.stride(1));
    }
    else assert(0);
  }
};


/// CUDA evaluator for generalized matrix-matrix products.
template <typename T1,
	  typename T2,
	  typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::gemp, be::cuda,
                 void(Block0&, T1, Block1 const&, Block2 const&, T2)>
{
  typedef typename Block0::value_type T;

  static bool const ct_valid = 
    impl::cuda::Traits<T>::valid &&
    impl::is_same<T, typename Block1::value_type>::value &&
    impl::is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block0>::value &&
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block0& r, T1, Block1 const& a, Block2 const& b, T2)
  {
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    dda::Data<Block0, dda::in> dda_r(r);
    dda::Data<Block1, dda::in> dda_a(a);
    dda::Data<Block2, dda::in> dda_b(b);

    // Blocks may not alias one another.
    bool is_not_aliased = 
      !impl::is_same_ptr(dda_r.ptr(), dda_b.ptr()) &&
      !impl::is_same_ptr(dda_r.ptr(), dda_a.ptr());

    bool is_a_row = impl::is_same<order1_type, row2_type>::value;
    bool is_b_row = impl::is_same<order2_type, row2_type>::value;

    return (is_not_aliased &&
	    dda_a.stride(is_a_row ? 1 : 0) == 1 &&
	    dda_b.stride(is_b_row ? 1 : 0) == 1);
  }

  static void exec(Block0& r, T1 alpha, Block1 const& a, 
		   Block2 const& b, T2 beta)
  {
    typedef typename Block0::value_type RT;

    typedef typename get_block_layout<Block0>::order_type order0_type;
    typedef typename get_block_layout<Block1>::order_type order1_type;
    typedef typename get_block_layout<Block2>::order_type order2_type;

    impl::cuda::dda::Data<Block0, dda::inout> dev_r(r);
    impl::cuda::dda::Data<Block1, dda::in> dev_a(a);
    impl::cuda::dda::Data<Block2, dda::in> dev_b(b);

    if (impl::is_same<order0_type, row2_type>::value)
    {
      bool is_a_row = impl::is_same<order1_type, row2_type>::value;
      char transa   = is_a_row ? 'n' : 't';
      int  lda      = is_a_row ? dev_a.stride(0) : dev_a.stride(1);

      bool is_b_row = impl::is_same<order2_type, row2_type>::value;
      char transb   = is_b_row ? 'n' : 't';
      int  ldb      = is_b_row ? dev_b.stride(0) : dev_b.stride(1);

      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)
      // to evaluate row-major matrix result with BLAS.

      impl::cuda::gemm(transb, transa,
		       b.size(2, 1),  // N
		       a.size(2, 0),  // M
		       a.size(2, 1),  // K
		       alpha,         // alpha
		       dev_b.ptr(), ldb,
		       dev_a.ptr(), lda,
		       beta,          // beta
		       dev_r.ptr(), dev_r.stride(0));
    }
    else if (impl::is_same<order0_type, col2_type>::value)
    {
      bool is_a_col = impl::is_same<order1_type, col2_type>::value;
      char transa   = is_a_col ? 'n' : 't';
      int  lda      = is_a_col ? dev_a.stride(1) : dev_a.stride(0);

      bool is_b_col = impl::is_same<order2_type, col2_type>::value;
      char transb   = is_b_col ? 'n' : 't';
      int  ldb      = is_b_col ? dev_b.stride(1) : dev_b.stride(0);

      impl::cuda::gemm(transa, transb,
		       a.size(2, 0),  // M
		       b.size(2, 1),  // N
		       a.size(2, 1),  // K
		       alpha,         // alpha
		       dev_a.ptr(), lda,
		       dev_b.ptr(), ldb,
		       beta,          // beta
		       dev_r.ptr(), dev_r.stride(1));
    }
    else assert(0);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_MATVEC_HPP
