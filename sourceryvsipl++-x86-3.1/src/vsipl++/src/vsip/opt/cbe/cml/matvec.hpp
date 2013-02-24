/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/cml/matvec.hpp
    @author  Don McCoy
    @date    2008-05-07
    @brief   VSIPL++ Library: CML matrix product evaluators.
*/

#ifndef VSIP_OPT_CBE_CML_MATVEC_HPP
#define VSIP_OPT_CBE_CML_MATVEC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/cbe/cml/prod.hpp>
#include <vsip/opt/cbe/cml/traits.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher 
{

/// CML evaluator for vector dot products (non-conjugated)
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::cml,
                 T(Block0 const&, Block1 const&)>
{
  static storage_format_type const complex1_format = get_block_layout<Block0>::storage_format;
  static storage_format_type const complex2_format = get_block_layout<Block1>::storage_format;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    // check that all data types are equal
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    // check complex layout is consistent
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value;

  static bool rt_valid(Block0 const&, Block1 const&) { return true; }

  static T exec(Block0 const& a, Block1 const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    dda::Data<Block0, dda::in> a_data(a);
    dda::Data<Block1, dda::in> b_data(b);

    T r = T();
    impl::cml::dot(a_data.ptr(), a_data.stride(0),
		   b_data.ptr(), b_data.stride(0),
		   &r, a.size(1, 0));
    return r;
  }
};


/// CML evaluator for vector dot products (conjugated)
template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::cml,
                 std::complex<T>(Block0 const&, 
				 expr::Unary<expr::op::Conj, Block1, true> const&)>
{
  static storage_format_type const complex1_format = get_block_layout<Block0>::storage_format;
  static storage_format_type const complex2_format = get_block_layout<Block1>::storage_format;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    // check that types are complex
    impl::is_complex<typename Block0::value_type>::value &&
    impl::is_complex<typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    // check complex layout is consistent
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value;

  static bool rt_valid(Block0 const&, 
		       expr::Unary<expr::op::Conj, Block1, true> const&)
  { return true;}

  static complex<T> exec(Block0 const& a, 
			 expr::Unary<expr::op::Conj, Block1, true> const& b)
  {
    assert(a.size(1, 0) == b.size(1, 0));

    dda::Data<Block0, dda::in> a_data(a);
    dda::Data<Block1, dda::in> b_data(b.arg());

    complex<T> r = complex<T>();
    impl::cml::dotc(a_data.ptr(), a_data.stride(0),
		    b_data.ptr(), b_data.stride(0),
		    &r, a.size(1, 0));
    return r;
  }
};


/// CML evaluator for outer products
template <typename T1,
          typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::outer, be::cml,
                 void(Block0&, T1, Block1 const&, Block2 const&)>
{
  typedef typename get_block_layout<Block0>::order_type order0_type;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    impl::cml::Cml_supports_block<Block2>::valid &&
    // check that the output is row-major 
    is_same<order0_type, row2_type>::value &&
    // check that all data types are equal
    is_same<T1, typename Block0::value_type>::value &&
    is_same<T1, typename Block1::value_type>::value &&
    is_same<T1, typename Block2::value_type>::value &&
    // check that the complex layouts are equal
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value &&
    impl::is_split_block<Block0>::value == impl::is_split_block<Block2>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& r, T1, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    bool unit_stride =
      (r_data.stride(1) == 1) &&
      (a_data.stride(0) == 1) && 
      (b_data.stride(0) == 1);

    return unit_stride;
  }

  static void exec(Block0& r, T1 alpha, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == r.size(2, 0));
    assert(b.size(1, 0) == r.size(2, 1));

    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // CML does not support a scaling parameter, so it is built into the
    // wrapper function.
    impl::cml::outer(alpha, 
		     a_data.ptr(), a_data.stride(0),
		     b_data.ptr(), b_data.stride(0),
		     r_data.ptr(), r_data.stride(0),
		     a.size(1, 0), b.size(1, 0));
  }
};


template <typename T1,
          typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::outer, be::cml,
                 void(Block0&, std::complex<T1>, Block1 const&, Block2 const&)>
{
  typedef typename get_block_layout<Block0>::order_type order0_type;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    impl::cml::Cml_supports_block<Block2>::valid &&
    // check that the output is row-major 
    is_same<order0_type, row2_type>::value &&
    // check that all data types are equal
    is_same<std::complex<T1>, typename Block0::value_type>::value &&
    is_same<std::complex<T1>, typename Block1::value_type>::value &&
    is_same<std::complex<T1>, typename Block2::value_type>::value &&
    // check that the complex layouts are equal
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value &&
    impl::is_split_block<Block0>::value == impl::is_split_block<Block2>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& r, std::complex<T1>, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    bool unit_stride =
      (r_data.stride(1) == 1) &&
      (a_data.stride(0) == 1) && 
      (b_data.stride(0) == 1);

    return unit_stride;
  }

  static void exec(Block0& r, std::complex<T1> alpha, Block1 const& a, Block2 const& b)
  {
    assert(a.size(1, 0) == r.size(2, 0));
    assert(b.size(1, 0) == r.size(2, 1));

    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // CML does not support a scaling parameter, so it is built into the
    // wrapper function.
    impl::cml::outer(alpha, 
		     a_data.ptr(), a_data.stride(0),
		     b_data.ptr(), b_data.stride(0),
		     r_data.ptr(), r_data.stride(0),
		     a.size(1, 0), b.size(1, 0));
  }
};


/// CML evaluator for matrix-vector products
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cml,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 1>::type>
{
  typedef typename Block0::value_type T;
  typedef typename get_block_layout<Block1>::order_type order1_type;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    impl::cml::Cml_supports_block<Block2>::valid &&
    // check that all data types are equal
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that the complex layouts are equal
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value &&
    impl::is_split_block<Block0>::value == impl::is_split_block<Block2>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& /*r*/, Block1 const& a, Block2 const& /*b*/)
  {
    dda::Data<Block1, dda::in> a_data(a);

    // For 'a', the dimension with the smallest stride must be one,
    // which depends on whether it is row- or column-major.
    bool is_a_row = is_same<order1_type, row2_type>::value;
    stride_type a_stride = is_a_row ? a_data.stride(1) : a_data.stride(0);

    return (a_stride == 1);
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // Either row- or column-major layouts are supported for the input 
    // matrix by using the identity:
    //   trans(r) = trans(b) * trans(a)
    // or just
    //   r = b * trans(a)  (since r and b are vectors)
    if (is_same<order1_type, row2_type>::value)
    {
      impl::cml::mvprod(a_data.ptr(), a_data.stride(0),
			b_data.ptr(), b_data.stride(0),
			r_data.ptr(), r_data.stride(0),
			a.size(2, 0),  // M
			a.size(2, 1)); // N
    }
    else if (is_same<order1_type, col2_type>::value)
    {
      impl::cml::vmprod(b_data.ptr(), b_data.stride(0),
			a_data.ptr(), a_data.stride(1),
			r_data.ptr(), r_data.stride(0),
			a.size(2, 1),  // N
			a.size(2, 0)); // M
    }
    else
      assert(0);
  }
};


/// CML evaluator for vector-matrix products
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cml,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 1 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;
  typedef typename get_block_layout<Block2>::order_type order2_type;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    impl::cml::Cml_supports_block<Block2>::valid &&
    // check that all data types are equal
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that the complex layouts are equal
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value &&
    impl::is_split_block<Block0>::value == impl::is_split_block<Block2>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& /*r*/, Block1 const& /*a*/, Block2 const& b)
  {
    dda::Data<Block2, dda::in> b_data(b);

    // For 'b', the dimension with the smallest stride must be one,
    // which depends on whether it is row- or column-major.
    bool is_b_row = is_same<order2_type, row2_type>::value;
    stride_type b_stride = is_b_row ? b_data.stride(1) : b_data.stride(0);

    return (b_stride == 1);
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // Either row- or column-major layouts are supported for the input 
    // matrix by using the identity:
    //   trans(r) = trans(b) * trans(a)
    // or just
    //   r = b * trans(a)  (since r and b are vectors)
    if (is_same<order2_type, row2_type>::value)
    {
      impl::cml::vmprod(a_data.ptr(), a_data.stride(0),
			b_data.ptr(), b_data.stride(0),
			r_data.ptr(), r_data.stride(0),
			b.size(2, 0),  // M
			b.size(2, 1)); // N
    }
    else if (is_same<order2_type, col2_type>::value)
    {
      impl::cml::mvprod(b_data.ptr(), b_data.stride(1),
			a_data.ptr(), a_data.stride(0),
			r_data.ptr(), r_data.stride(0),
			b.size(2, 1),  // N
			b.size(2, 0)); // M
    }
    else
      assert(0);
  }
};


/// CML evaluator for matrix-matrix products
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cml,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if_c<Block1::dim == 2 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;
  typedef typename get_block_layout<Block0>::order_type order0_type;
  typedef typename get_block_layout<Block1>::order_type order1_type;
  typedef typename get_block_layout<Block2>::order_type order2_type;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    impl::cml::Cml_supports_block<Block2>::valid &&
    // check that all data types are equal
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that the complex layouts are equal
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value &&
    impl::is_split_block<Block0>::value == impl::is_split_block<Block2>::value &&
    // check that the layout is row-major for the first input and the output
    is_same<order0_type, row2_type>::value && 
    is_same<order1_type, row2_type>::value && 
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // For 'b', the dimension with the smallest stride must be one,
    // which depends on whether it is row- or column-major.
    bool is_b_row = is_same<order2_type, row2_type>::value;
    stride_type b_stride = is_b_row ? b_data.stride(1) : b_data.stride(0);

    return r_data.stride(1) == 1 && a_data.stride(1) == 1 && b_stride == 1;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // Either row- or column-major layouts are supported for
    // the second input by mapping them to the normal product
    // or transpose product respectively.
    if (is_same<order2_type, row2_type>::value)
    {
      impl::cml::mprod(a_data.ptr(), a_data.stride(0),
		       b_data.ptr(), b_data.stride(0),
		       r_data.ptr(), r_data.stride(0),
		       a.size(2, 0),  // M
		       a.size(2, 1),  // N
		       b.size(2, 1)); // P
    }
    else if (is_same<order2_type, col2_type>::value)
    {
      impl::cml::mprodt(a_data.ptr(), a_data.stride(0),
			b_data.ptr(), b_data.stride(1),
			r_data.ptr(), r_data.stride(0),
			a.size(2, 0),  // M
			a.size(2, 1),  // N
			b.size(2, 1)); // P
    }
    else
      assert(0);
  }
};


/// CML evaluator for matrix-matrix conjugate products
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prodj, be::cml,
                 void(Block0&, Block1 const&, Block2 const&)>
{
  typedef typename Block0::value_type T;
  typedef typename get_block_layout<Block0>::order_type order0_type;
  typedef typename get_block_layout<Block1>::order_type order1_type;
  typedef typename get_block_layout<Block2>::order_type order2_type;

  static bool const ct_valid = 
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<Block0>::valid &&
    impl::cml::Cml_supports_block<Block1>::valid &&
    impl::cml::Cml_supports_block<Block2>::valid &&
    // check that all data types are equal
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that the complex layouts are equal
    impl::is_split_block<Block0>::value == impl::is_split_block<Block1>::value &&
    impl::is_split_block<Block0>::value == impl::is_split_block<Block2>::value &&
    // check that the layout is row-major for the first input and the output
    is_same<order0_type, row2_type>::value && 
    is_same<order1_type, row2_type>::value && 
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& r, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // For 'b', the dimension with the smallest stride must be one,
    // which depends on whether it is row- or column-major.
    bool is_b_row = is_same<order2_type, row2_type>::value;
    stride_type b_stride = is_b_row ? b_data.stride(1) : b_data.stride(0);

    return r_data.stride(1) == 1 && a_data.stride(1) == 1 && b_stride == 1;
  }

  static void exec(Block0& r, Block1 const& a, Block2 const& b)
  {
    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    // Either row- or column-major layouts are supported for
    // the second input by mapping them to the normal product
    // or transpose product respectively.
    if (is_same<order2_type, row2_type>::value)
    {
      impl::cml::mprodj(a_data.ptr(), a_data.stride(0),
			b_data.ptr(), b_data.stride(0),
		        r_data.ptr(), r_data.stride(0),
			a.size(2, 0),  // M
			a.size(2, 1),  // N
			b.size(2, 1)); // P
    }
    else if (is_same<order2_type, col2_type>::value)
    {
      impl::cml::mprodh(a_data.ptr(), a_data.stride(0),
			b_data.ptr(), b_data.stride(1),
			r_data.ptr(), r_data.stride(0),
			a.size(2, 0),  // M
			a.size(2, 1),  // N
			b.size(2, 1)); // P
    }
    else
      assert(0);
  }
};


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
