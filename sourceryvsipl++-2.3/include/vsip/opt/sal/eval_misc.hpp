/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/eval_misc.hpp
    @author  Don McCoy
    @date    2005-10-17
    @brief   VSIPL++ Library: SAL evaluators (for use in general dispatch).

*/

#ifndef VSIP_OPT_SAL_EVAL_MISC_HPP
#define VSIP_OPT_SAL_EVAL_MISC_HPP

#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/sal/bindings.hpp>


/// This switch chooses between two sets of matrix-multiply functions
/// provided by SAL.  Setting to '1' will select the new sal::mat_mul()
/// variants and setting it to '0' will select the older sal::mmul() 
/// set.  At present, Mecury states that the mmul() functions are 
/// faster, but the API is changing towards that used with mat_mul().
/// See 'sal.hpp' for details as to the differences.
///
/// In addition, complex mat_mul variants have a defect that
/// affect CSAL and SAL for MCOE 6.3.0.
#ifndef VSIP_IMPL_SAL_USE_MAT_MUL
#define VSIP_IMPL_SAL_USE_MAT_MUL 0
#endif

namespace vsip
{
namespace impl
{
namespace sal
{
template <typename T>
struct Mul_inv
{
  typedef T result_type;
  static result_type apply(T value)
  {
    return T(1) / value;
  }
};
    

template <typename T>
struct Mul_inv<std::complex<T> >
{
  typedef std::complex<T> result_type;
  static result_type apply(std::complex<T> value)
  {
    return complex<T>(1, 0) / value;
  }
};

/// gives real or complex multiplicative inverse, depending on value type.
template <typename T>
inline
typename Mul_inv<T>::result_type
multiplicative_inverse(T value)
{
  return Mul_inv<T>::apply(value);
};

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

#if VSIP_IMPL_SAL_USE_MAT_MUL

/// Mercury SAL evaluator for outer product
template <typename T1,
          typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::outer, be::mercury_sal,
                 void(Block0&, T1, Block1 const&, Block2 const&)>
{
  static bool const ct_valid = 
    impl::sal::Sal_traits<T1>::valid &&
    impl::Type_equal<T1, typename Block1::value_type>::value &&
    impl::Type_equal<T1, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0;

  static bool rt_valid(Block0& r, T1, Block1 const& a, Block2 const& b)
  {
    typedef typename impl::Block_layout<Block0>::order_type order0_type;
    dimension_type const r_dim1 = order0_type::impl_dim1;

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    bool unit_stride =
      (ext_r.stride(r_dim1) == 1) &&
      (ext_a.stride(0) == 1) && 
      (ext_b.stride(0) == 1);

    return unit_stride;
  }

  static void exec(Block0& r, T1 alpha, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == r.size(2, 0));
    assert(b.size(1, 0) == r.size(2, 1));

    typedef typename impl::Block_layout<Block0>::order_type order0_type;

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order0_type, row2_type>::value)
    {
      impl::sal::mat_mul(a.size(1, 0), b.size(1, 0),
			 1,
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 0);
    }
    else if (impl::Type_equal<order0_type, col2_type>::value)
    {
      // Use identity:
      //   R = A B     <=>     trans(R) = trans(B) trans(A)

      impl::sal::mat_mul(b.size(1, 0), a.size(1, 0),
			 1,
			 ext_b.data(), ext_b.stride(0),
			 ext_a.data(), ext_a.stride(0),
			 ext_r.data(), ext_r.stride(1),
			 0);
    }
    else
      assert(0);

    // SAL does not support a scaling parameter
    impl::sal::vsmul(ext_r.data(), 1,
		     &alpha,
		     ext_r.data(), 1,
		     r.size());
  }
};


/// Mercury SAL evaluator for outer product (with conjugate)
template <typename T1,
          typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::outer, be::mercury_sal,
                 void(Block0&, std::complex<T1>, Block1 const&, Block2 const&)>
{
  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<std::complex<T1> >::valid &&
    impl::Type_equal<std::complex<T1>, typename Block1::value_type>::value &&
    impl::Type_equal<std::complex<T1>, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0& r, std::complex<T1>, 
		       Block1 const& a, Block2 const& b)
  {
    typedef typename impl::Block_layout<Block0>::order_type order0_type;
    dimension_type const r_dim1 = order0_type::impl_dim1;

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    bool unit_stride =
      (ext_r.stride(r_dim1) == 1) &&
      (ext_a.stride(0) == 1) && 
      (ext_b.stride(0) == 1);

    return unit_stride;
  }

  static void exec(Block0& r, std::complex<T1> alpha, 
		   Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == r.size(2, 0));
    assert(b.size(1, 0) == r.size(2, 1));

    typedef typename impl::Block_layout<Block0>::order_type order0_type;
    typedef typename impl::Block_layout<Block1>::order_type order1_type;
    typedef typename impl::Block_layout<Block2>::order_type order2_type;

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order0_type, row2_type>::value)
    {
      impl::sal::mat_mul(a.size(1, 0), b.size(1, 0),
			 1,
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 SAL_CONJUGATE_RIGHT);
    }
    else if (impl::Type_equal<order0_type, col2_type>::value)
    {
      // Use identity:
      //   R = A B     <=>     trans(R) = trans(B) trans(A)

      impl::sal::mat_mul(b.size(1, 0), a.size(1, 0),
			 1,
			 ext_b.data(), ext_b.stride(0),
			 ext_a.data(), ext_a.stride(0),
			 ext_r.data(), ext_r.stride(1),
			 SAL_CONJUGATE_LEFT);
    }
    else
      assert(0);

    // SAL does not support a scaling parameter
    impl::sal::vsmul(ext_r.data(), 1,
		     &alpha,
		     ext_r.data(), 1,
		     r.size());
  }
};

#endif // VSIP_IMPL_SAL_USE_MAT_MUL


/// Mercury SAL evaluator for vector-vector dot-product.
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::mercury_sal,
                 T(Block0 const&, Block1 const&)>
{
  typedef typename impl::Block_layout<Block0>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    // check complex layout is consistent
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0 const&, Block1 const&) { return true; }

  static T exec(Block0 const& a, Block1 const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    impl::Ext_data<Block0> ext_a(const_cast<Block0&>(a));
    impl::Ext_data<Block1> ext_b(const_cast<Block1&>(b));

    T r = impl::sal::dot(a.size(1, 0),
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0));
    return r;
  }
};

/// SAL evaluator for vector-vector dot-product (conjugated).
template <typename T,
          typename Block1,
          typename Block2>
struct Evaluator<op::dot, be::mercury_sal,
                 std::complex<T>(Block1 const&, 
				 expr::Unary<expr::op::Conj, Block2, true> const&)>
{
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<complex<T> >::valid &&
    impl::Type_equal<complex<T>, typename Block1::value_type>::value &&
    impl::Type_equal<complex<T>, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check complex layout is consistent
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block1 const&, 
		       expr::Unary<expr::op::Conj, Block2, true> const&)
  { return true;}

  static complex<T> exec(Block1 const& a, 
			 expr::Unary<expr::op::Conj, Block2, true> const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b.arg()));

    return impl::sal::dotc(a.size(1, 0),
			   ext_b.data(), ext_b.stride(0),
			   ext_a.data(), ext_a.stride(0) );
    // Note:
    //   SAL     cidotprx(x, y)  => conj(x) * y, while 
    //   VSIPL++ cvjdot(x, y) => x * conj(y)
  }
};



#if VSIP_IMPL_SAL_USE_MAT_MUL

/// Mercury SAL evaluator for matrix-vector products.
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::mercury_sal,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 1>::type>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    dimension_type const a_dim0 = order1_type::impl_dim0;
    dimension_type const a_dim1 = order1_type::impl_dim1;

    bool unit_minor_stride = 
      (ext_a.stride(a_dim0) == ext_a.size(a_dim1) * ext_a.stride(a_dim1)) &&
      (ext_r.stride(0) == 1) && 
      (ext_b.stride(0) == 1);

    return unit_minor_stride;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(2, 1) == b.size(1, 0));

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order1_type, row2_type>::value)
    {
      impl::sal::mat_mul(a.size(2, 0), 1,
			 a.size(2, 1),
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 0);
    }
    else if (impl::Type_equal<order1_type, col2_type>::value)
    {
      impl::sal::mat_mul(1, a.size(2, 0),
			 a.size(2, 1),
			 ext_b.data(), ext_b.stride(0),
			 ext_a.data(), ext_a.stride(1),
			 ext_r.data(), ext_r.stride(0),
			 0);
    }
    else
      assert(0);
  }
};


/// Mercury SAL evaluator for vector-matrix products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::mercury_sal,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 1 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    dimension_type const b_dim0 = order2_type::impl_dim0;
    dimension_type const b_dim1 = order2_type::impl_dim1;

    bool unit_minor_stride = 
      (ext_b.stride(b_dim0) == ext_b.size(b_dim1) * ext_b.stride(b_dim1)) &&
      (ext_r.stride(0) == 1) && 
      (ext_a.stride(0) == 1);

    return unit_minor_stride;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == b.size(2, 0));

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order2_type, row2_type>::value)
    {
      impl::sal::mat_mul(1, b.size(2, 1),
			 b.size(2, 0),
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 0);
    }
    else if (impl::Type_equal<order2_type, col2_type>::value)
    {
      impl::sal::mat_mul(b.size(2, 1), 1,
			 b.size(2, 0),
			 ext_b.data(), ext_b.stride(1),
			 ext_a.data(), ext_a.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 0);
    }
    else 
      assert(0);
  }
};


/// Mercury SAL evaluator for matrix-matrix products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::mercury_sal,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check dimension ordering
    order0_type::impl_dim0 == order1_type::impl_dim0 &&
    order1_type::impl_dim0 == order2_type::impl_dim0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    dimension_type const r_dim0 = order0_type::impl_dim0;
    dimension_type const r_dim1 = order0_type::impl_dim1;
    dimension_type const a_dim0 = order1_type::impl_dim0;
    dimension_type const a_dim1 = order1_type::impl_dim1;
    dimension_type const b_dim0 = order2_type::impl_dim0;
    dimension_type const b_dim1 = order2_type::impl_dim1;

    bool unit_minor_stride =
      (ext_r.stride(r_dim1) == 1) &&
      (ext_a.stride(a_dim1) == 1) &&
      (ext_b.stride(b_dim1) == 1);

    return unit_minor_stride;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(2, 0) == r.size(2, 0));
    assert(a.size(2, 1) == b.size(2, 0));
    assert(b.size(2, 1) == r.size(2, 1));

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order0_type, row2_type>::value)
    {
      impl::sal::mat_mul(a.size(2, 0), b.size(2, 1),
			 b.size(2, 0),
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 0);
    }
    else if (impl::Type_equal<order0_type, col2_type>::value)
    {
      // use r = a b  ==> trans(r) = trans(b) trans(a)

      impl::sal::mat_mul(b.size(2, 1), a.size(2, 0), 
			 b.size(2, 0), 
			 ext_b.data(), ext_b.stride(1),
			 ext_a.data(), ext_a.stride(1),
			 ext_r.data(), ext_r.stride(1),
			 0);
    }
    else assert(0);
  }

};



#else // !VSIP_IMPL_SAL_USE_MAT_MUL

/// Mercury SAL evaluator for matrix-vector products.
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::mercury_sal,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 1>::type>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0&, Block1 const& a, Block2 const&)
  {
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));

    dimension_type const a_dim0 = order1_type::impl_dim0;
    dimension_type const a_dim1 = order1_type::impl_dim1;

    bool dense_major_dim = 
      (ext_a.stride(a_dim0) == (stride_type)ext_a.size(a_dim1) *
                                            ext_a.stride(a_dim1));

    return dense_major_dim;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(2, 1) == b.size(1, 0));

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order1_type, row2_type>::value)
    {
      impl::sal::mmul(a.size(2, 0), // M
		      1,            // N
		      a.size(2, 1), // P
		      ext_a.data(), ext_a.stride(1),
		      ext_b.data(), ext_b.stride(0),
		      ext_r.data(), ext_r.stride(0));
    }
    else if (impl::Type_equal<order1_type, col2_type>::value)
    {
      impl::sal::mmul(1,            // M
		      a.size(2, 0), // N
		      a.size(2, 1), // P
		      ext_b.data(), ext_b.stride(0),
		      ext_a.data(), ext_a.stride(0),
		      ext_r.data(), ext_r.stride(0));
    }
    else 
      assert(0);
  }
};


/// Mercury SAL evaluator for vector-matrix products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::mercury_sal,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 1 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0&, Block1 const&, Block2 const& b)
  {
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    dimension_type const b_dim0 = order2_type::impl_dim0;
    dimension_type const b_dim1 = order2_type::impl_dim1;

    bool dense_major_dim = 
      (ext_b.stride(b_dim0) == (stride_type)ext_b.size(b_dim1) *
					    ext_b.stride(b_dim1));

    return dense_major_dim;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == b.size(2, 0));

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order2_type, row2_type>::value)
    {
      impl::sal::mmul(1,            // M
		      b.size(2, 1), // N
		      b.size(2, 0), // P
		      ext_a.data(), ext_a.stride(0),
		      ext_b.data(), ext_b.stride(1),
		      ext_r.data(), ext_r.stride(0));
    }
    else if (impl::Type_equal<order2_type, col2_type>::value)
    {
      impl::sal::mmul(b.size(2, 1), // M
		      1,            // N
		      b.size(2, 0), // P
		      ext_b.data(), ext_b.stride(0),
		      ext_a.data(), ext_a.stride(0),
		      ext_r.data(), ext_r.stride(0));
    }
    else 
      assert(0);
  }
};


/// Mercury SAL evaluator for matrix-matrix products.
template <typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::prod, be::mercury_sal,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check dimension ordering
    order0_type::impl_dim0 == order1_type::impl_dim0 &&
    order1_type::impl_dim0 == order2_type::impl_dim0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;


  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    dimension_type const r_dim0 = order0_type::impl_dim0;
    dimension_type const r_dim1 = order0_type::impl_dim1;
    dimension_type const a_dim0 = order1_type::impl_dim0;
    dimension_type const a_dim1 = order1_type::impl_dim1;
    dimension_type const b_dim0 = order2_type::impl_dim0;
    dimension_type const b_dim1 = order2_type::impl_dim1;

    bool dense_major_dim = 
      (ext_r.stride(r_dim0) == (stride_type)ext_r.size(r_dim1) *
                                            ext_r.stride(r_dim1)) &&
      (ext_a.stride(a_dim0) == (stride_type)ext_a.size(a_dim1) *
                                            ext_a.stride(a_dim1)) &&
      (ext_b.stride(b_dim0) == (stride_type)ext_b.size(b_dim1) *
                                            ext_b.stride(b_dim1));

    return dense_major_dim;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    assert(a.size(2, 0) == r.size(2, 0));
    assert(a.size(2, 1) == b.size(2, 0));
    assert(b.size(2, 1) == r.size(2, 1));

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    if (impl::Type_equal<order0_type, row2_type>::value)
    {
      impl::sal::mmul(a.size(2, 0), // M
		      b.size(2, 1), // N
		      a.size(2, 1), // P
		      ext_a.data(), ext_a.stride(1),
		      ext_b.data(), ext_b.stride(1),
		      ext_r.data(), ext_r.stride(1));
    }
    else if (impl::Type_equal<order0_type, col2_type>::value)
    {
      // use r = a b  ==> trans(r) = trans(b) trans(a)

      impl::sal::mmul(b.size(2, 1), // M
		      a.size(2, 0), // N
		      b.size(2, 0), // P
		      ext_b.data(), ext_b.stride(0),
		      ext_a.data(), ext_a.stride(0),
		      ext_r.data(), ext_r.stride(0));
    }
    else assert(0);
  }
};

#endif

#if VSIP_IMPL_SAL_USE_MAT_MUL

/// Mercury SAL evaluator for general product
template <typename T1,
	  typename T2,
	  typename Block0,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::gemp, be::mercury_sal,
                 void(Block0&, T1, Block1 const&, Block2 const&, T2)>
{
  typedef typename Block0::value_type T;

  typedef typename impl::Block_layout<Block0>::order_type order0_type;
  typedef typename impl::Block_layout<Block1>::order_type order1_type;
  typedef typename impl::Block_layout<Block2>::order_type order2_type;

  typedef typename impl::Block_layout<Block0>::complex_type complex0_type;
  typedef typename impl::Block_layout<Block1>::complex_type complex1_type;
  typedef typename impl::Block_layout<Block2>::complex_type complex2_type;

  static bool const ct_valid = 
    impl::sal::Sal_traits<T>::valid &&
    impl::Type_equal<T, typename Block0::value_type>::value &&
    impl::Type_equal<T, typename Block1::value_type>::value &&
    impl::Type_equal<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    impl::Ext_data_cost<Block0>::value == 0 &&
    impl::Ext_data_cost<Block1>::value == 0 &&
    impl::Ext_data_cost<Block2>::value == 0 &&
    // check dimension ordering
    order0_type::impl_dim0 == order1_type::impl_dim0 &&
    order1_type::impl_dim0 == order2_type::impl_dim0 &&
    // check complex layout
    impl::Type_equal<complex0_type, complex1_type>::value &&
    impl::Type_equal<complex1_type, complex2_type>::value;

  static bool rt_valid(Block0& r, T1, Block1 const& a, 
		       Block2 const& b, T2)
  {
    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    // dimension_type const r_dim0 = order0_type::impl_dim0;
    dimension_type const r_dim1 = order0_type::impl_dim1;
    // dimension_type const a_dim0 = order1_type::impl_dim0;
    dimension_type const a_dim1 = order1_type::impl_dim1;
    // dimension_type const b_dim0 = order2_type::impl_dim0;
    dimension_type const b_dim1 = order2_type::impl_dim1;

    bool unit_minor_stride =
      (ext_r.stride(r_dim1) == 1) &&
      (ext_a.stride(a_dim1) == 1) &&
      (ext_b.stride(b_dim1) == 1);

    return unit_minor_stride;
  }

  static void exec(Block0& r, T1 alpha, Block1 const& a, 
		   Block2 const& b, T2 beta)
  {
    typedef typename Block0::value_type RT;

    typedef typename impl::Block_layout<Block0>::order_type order0_type;
    typedef typename impl::Block_layout<Block1>::order_type order1_type;
    typedef typename impl::Block_layout<Block2>::order_type order2_type;

    impl::Ext_data<Block0> ext_r(const_cast<Block0&>(r));
    impl::Ext_data<Block1> ext_a(const_cast<Block1&>(a));
    impl::Ext_data<Block2> ext_b(const_cast<Block2&>(b));

    // to implement C = alpha A B + beta C, the scaling factors
    // must be implemented manually

    impl::sal::vsmul(ext_r.data(), 1,
		     &beta,
		     ext_r.data(), 1,
		     r.size());

    // we must fold the scaling into this prior to calling the function
    // because of the accumulation option, this must be undone prior
    // to returning
    impl::sal::vsmul(ext_a.data(), 1,
		     &alpha,
		     ext_a.data(), 1,
		     a.size());


    if (impl::Type_equal<order0_type, row2_type>::value)
    {
      impl::sal::mat_mul(a.size(2, 0), b.size(2, 1),
			 b.size(2, 0),
			 ext_a.data(), ext_a.stride(0),
			 ext_b.data(), ext_b.stride(0),
			 ext_r.data(), ext_r.stride(0),
			 SAL_ACCUMULATE);
    }
    else if (impl::Type_equal<order0_type, col2_type>::value)
    {
      // Use identity:
      //   R = A B      <=>     trans(R) = trans(B) trans(A)

      impl::sal::mat_mul(b.size(2, 1), a.size(2, 0), 
			 b.size(2, 0), 
			 ext_b.data(), ext_b.stride(1),
			 ext_a.data(), ext_a.stride(1),
			 ext_r.data(), ext_r.stride(1),
			 SAL_ACCUMULATE);
    }
    else assert(0);


    // undo the scaling of alpha
    T1 one_over_alpha = impl::sal::multiplicative_inverse(alpha);

    impl::sal::vsmul(ext_a.data(), 1,
		     &one_over_alpha,
		     ext_a.data(), 1,
		     a.size());
  }
};

#endif // VSIP_IMPL_SAL_USE_MAT_MUL

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_SAL_EVAL_MISC_HPP
