//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_matvec_hpp_
#define ovxx_cvsip_matvec_hpp_

#include <ovxx/config.hpp>
#include <ovxx/order_traits.hpp>
#include <ovxx/cvsip/block.hpp>
#include <ovxx/cvsip/view.hpp>

namespace ovxx
{
namespace cvsip
{

template <typename T> struct op_traits { static bool const valid = false;};

#if OVXX_CVSIP_HAVE_FLOAT
template <>
struct op_traits<float>
{
  typedef float value_type;
  typedef vsip_vview_f vector_type;
  typedef vsip_mview_f matrix_type;

  static bool const valid = true;

  static value_type dot(vector_type const *a, vector_type const *b)
  { return vsip_vdot_f(a, b);}
  static void outer(value_type s, vector_type const *a, vector_type const *b,
                    matrix_type *r)
  { vsip_vouter_f(s, a, b, r);}
  static void gemp(matrix_type *c, value_type alpha,
                   matrix_type *a, matrix_type *b, value_type beta)
  { vsip_gemp_f(alpha, a, VSIP_MAT_NTRANS, b, VSIP_MAT_NTRANS, beta, c);}
  static void prod(matrix_type *a, matrix_type *b, matrix_type *c)
  { vsip_mprod_f(a, b, c);}
  static void prod(matrix_type *a, vector_type *x, vector_type *y)
  { vsip_mvprod_f(a, x, y);}
};

template <>
struct op_traits<std::complex<float> >
{
  typedef std::complex<float> value_type;
  typedef vsip_cvview_f vector_type;
  typedef vsip_cmview_f matrix_type;

  static bool const valid = true;

  static value_type dot(vector_type const *a, vector_type const *b)
  {
    vsip_cscalar_f retn = vsip_cvdot_f(a, b);
    return value_type(retn.r, retn.i);
  }
  static value_type cvjdot(vector_type const *a, vector_type const *b)
  {
    vsip_cscalar_f retn = vsip_cvjdot_f(a, b);
    return value_type(retn.r, retn.i);
  }
  static void outer(value_type s, vector_type const *a, vector_type const *b,
                    matrix_type *r)
  {
    vsip_cscalar_f scale = {s.real(), s.imag()};
    vsip_cvouter_f(scale, a, b, r);
  }
  static void gemp(matrix_type *c, value_type alpha,
                   matrix_type *a, matrix_type *b, value_type beta)
  {
    vsip_cscalar_f al = {alpha.real(), alpha.imag()};
    vsip_cscalar_f be = {beta.real(), beta.imag()};
    vsip_cgemp_f(al, a, VSIP_MAT_NTRANS, b, VSIP_MAT_NTRANS, be, c);
  }
  static void prod(matrix_type *a, matrix_type *b, matrix_type *c)
  { vsip_cmprod_f(a, b, c);}
  static void prod(matrix_type *a, vector_type *x, vector_type *y)
  { vsip_cmvprod_f(a, x, y);}
};
#endif
#if OVXX_CVSIP_HAVE_DOUBLE
template <>
struct op_traits<double>
{
  typedef double value_type;
  typedef vsip_vview_d vector_type;
  typedef vsip_mview_d matrix_type;

  static bool const valid = true;

  static value_type dot(vector_type const *a, vector_type const *b)
  { return vsip_vdot_d(a, b);}
  static void outer(value_type s, vector_type const *a, vector_type const *b,
                    matrix_type *r)
  { vsip_vouter_d(s, a, b, r);}
  static void gemp(matrix_type *c, value_type alpha,
                   matrix_type *a, matrix_type *b, value_type beta)
  { vsip_gemp_d(alpha, a, VSIP_MAT_NTRANS, b, VSIP_MAT_NTRANS, beta, c);}
  static void prod(matrix_type *a, matrix_type *b, matrix_type *c)
  { vsip_mprod_d(a, b, c);}
  static void prod(matrix_type *a, vector_type *x, vector_type *y)
  { vsip_mvprod_d(a, x, y);}
};

template <>
struct op_traits<std::complex<double> >
{
  typedef std::complex<double> value_type;
  typedef vsip_cvview_d vector_type;
  typedef vsip_cmview_d matrix_type;

  static bool const valid = true;

  static value_type dot(vector_type const *a, vector_type const *b)
  {
    vsip_cscalar_d retn = vsip_cvdot_d(a, b);
    return value_type(retn.r, retn.i);
  }  
  static value_type cvjdot(vector_type const *a, vector_type const *b)
  {
    vsip_cscalar_d retn = vsip_cvjdot_d(a, b);
    return value_type(retn.r, retn.i);
  }  
  static void outer(value_type s, vector_type const *a, vector_type const *b,
                    matrix_type *r)
  {
    vsip_cscalar_d scale = {s.real(), s.imag()};
    vsip_cvouter_d(scale, a, b, r);
  }
  static void gemp(matrix_type *c, value_type alpha,
                   matrix_type *a, matrix_type *b, value_type beta)
  {
    vsip_cscalar_d al = {alpha.real(), alpha.imag()};
    vsip_cscalar_d be = {beta.real(), beta.imag()};
    vsip_cgemp_d(al, a, VSIP_MAT_NTRANS, b, VSIP_MAT_NTRANS, be, c);
  }
  static void prod(matrix_type *a, matrix_type *b, matrix_type *c)
  { vsip_cmprod_d(a, b, c);}
  static void prod(matrix_type *a, vector_type *x, vector_type *y)
  { vsip_cmvprod_d(a, x, y);}
};
#endif

} // namespace ovxx::cvsip

namespace dispatcher
{

template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::cvsip,
                 T(Block0 const&, Block1 const&)>
{
  typedef cvsip::op_traits<T> traits;

  // Note: C-VSIPL backends set ct_valid to false if the block types don't 
  // allow direct data access (i.e. dda::Data<>::ct_cost > 0), which occurs when
  // an operator like conj() is used (see cvjdot()).  This prevents this
  // evaluator from being used if invoked through the normal dispatch 
  // mechanism.

  static bool const ct_valid = 
    traits::valid &&
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0 const&, Block1 const&) { return true;}

  static T exec(Block0 const& a, Block1 const& b)
  {
    namespace c = cvsip;
    OVXX_PRECONDITION(a.size(1, 0) == b.size(1, 0));

    dda::Data<Block0, dda::in> a_data(a);
    dda::Data<Block1, dda::in> b_data(b);
    c::const_View<1, T> aview(a_data.ptr(), 0, a_data.stride(0), a.size(1, 0));
    c::const_View<1, T> bview(b_data.ptr(), 0, b_data.stride(0), b.size(1, 0));
    return traits::dot(aview.ptr(), bview.ptr());
  }
};


template <typename T,
          typename Block0,
          typename Block1>
struct Evaluator<op::dot, be::cvsip,
                 std::complex<T>(Block0 const&, 
				 expr::Unary<expr::op::Conj, Block1, true> const&)>
{
  typedef cvsip::op_traits<std::complex<T> > traits;
  typedef expr::Unary<expr::op::Conj, Block1, true> block1_type;

  static bool const ct_valid = 
    traits::valid &&
    is_same<complex<T>, typename Block0::value_type>::value &&
    is_same<complex<T>, typename Block1::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::in>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0 const&, block1_type const&)
  { return true; }

  static complex<T> exec(Block0 const& a, block1_type const& b)
  {
    namespace c = cvsip;
    OVXX_PRECONDITION(a.size(1, 0) == b.size(1, 0));
    dda::Data<Block0, dda::in> a_data(a);
    dda::Data<Block1, dda::in> b_data(b.arg());
    c::const_View<1, std::complex<T> >
      aview(a_data.ptr(), 0, a_data.stride(0), a.size(1, 0));
    c::const_View<1, std::complex<T> >
      bview(b_data.ptr(), 0, b_data.stride(0), b.size(1, 0));
    return traits::cvjdot(aview.ptr(), bview.ptr());
  }
};

template <typename T,
          typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::outer, be::cvsip,
                 void(Block0&, T, Block1 const&, Block2 const&)>
{
  typedef cvsip::op_traits<T> traits;

  static bool const ct_valid = 
    traits::valid &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0&, T, Block1 const&, Block2 const&)
  { return true;}

  static void exec(Block0& r, T s, Block1 const& a, Block2 const& b)
  {
    namespace c = cvsip;
    typedef typename get_block_layout<Block0>::order_type order0_type;

    OVXX_PRECONDITION(a.size(1, 0) == r.size(2, 0));
    OVXX_PRECONDITION(b.size(1, 0) == r.size(2, 1));

    dda::Data<Block0, dda::out> r_data(r);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    c::const_View<1, T> aview(a_data.ptr(), 0, a_data.stride(0), a.size(1, 0));
    c::const_View<1, T> bview(b_data.ptr(), 0, b_data.stride(0), b.size(1, 0));
    c::View<2, T> rview(r_data.ptr(), 0,
			r_data.stride(0), r.size(2, 0),
			r_data.stride(1), r.size(2, 1));

    traits::outer(s, aview.ptr(), bview.ptr(), rview.ptr());
  }
};

/// Vector-Matrix product
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cvsip,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if<Block1::dim == 1 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;
  typedef cvsip::op_traits<T> traits;

  static bool const ct_valid = 
    traits::valid &&
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0& y, Block1 const& x, Block2 const& a)
  { return true;}

  static void exec(Block0& y, Block1 const& x, Block2 const& a)
  {
    namespace c = cvsip;
    OVXX_PRECONDITION(x.size(1, 0) == a.size(2, 0));

    dda::Data<Block0, dda::out> y_data(y);
    dda::Data<Block1, dda::in> x_data(x);
    dda::Data<Block2, dda::in> a_data(a);

    c::View<1, T> yview(y_data.ptr(), 0, y_data.stride(0), y.size(1, 0));
    c::const_View<1, T> xview(x_data.ptr(), 0, x_data.stride(0), x.size(1, 0));
    c::const_View<2, T> aview(a_data.ptr(), 0,
			      a_data.stride(1), a.size(2, 1),
			      a_data.stride(0), a.size(2, 0));
    
    traits::prod(aview.ptr(), xview.ptr(), yview.ptr());
  }
};

/// Matrix-vector product.
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cvsip, 
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if<Block1::dim == 2 && Block2::dim == 1>::type>
{
  typedef typename Block0::value_type T;
  typedef typename get_block_layout<Block1>::order_type order1_type;
  typedef cvsip::op_traits<T> traits;

  static bool const ct_valid = 
    traits::valid &&
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0&, Block1 const&, Block2 const&)
  { return true;}

  static void exec(Block0& y, Block1 const& a, Block2 const& x)
  {
    namespace c = cvsip;
    OVXX_PRECONDITION(x.size(1, 0) == a.size(2, 1));

    dda::Data<Block0, dda::out> y_data(y);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> x_data(x);

    c::View<1, T> yview(y_data.ptr(), 0, y_data.stride(0), y.size(1, 0));
    c::const_View<2, T> aview(a_data.ptr(), 0,
			      a_data.stride(0), a.size(2, 0),
			      a_data.stride(1), a.size(2, 1));
    c::const_View<1, T> xview(x_data.ptr(), 0, x_data.stride(0), x.size(1, 0));

    traits::prod(aview.ptr(), xview.ptr(), yview.ptr());
  }
};

/// Matrix-matrix product
template <typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::prod, be::cvsip,
                 void(Block0&, Block1 const&, Block2 const&),
		 typename enable_if<Block1::dim == 2 && Block2::dim == 2>::type>
{
  typedef typename Block0::value_type T;
  typedef cvsip::op_traits<T> traits;

  static bool const ct_valid = 
    traits::valid &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::out>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0&, Block1 const&, Block2 const&)
  { return true;}

  static void exec(Block0& c, Block1 const& a, Block2 const& b)
  {
    namespace cc = cvsip;
    dda::Data<Block0, dda::out> c_data(c);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    cc::View<2, T> cview(c_data.ptr(), 0,
			 c_data.stride(0), c.size(2, 0),
			 c_data.stride(1), c.size(2, 1));
    cc::const_View<2, T> aview(a_data.ptr(), 0,
			       a_data.stride(0), a.size(2, 0),
			       a_data.stride(1), a.size(2, 1));
    cc::const_View<2, T> bview(b_data.ptr(), 0,
			       b_data.stride(0), b.size(2, 0),
			       b_data.stride(1), b.size(2, 1));
    traits::prod(aview.ptr(), bview.ptr(), cview.ptr());
  }
};

template <typename T,
          typename Block0,
          typename Block1,
          typename Block2>
struct Evaluator<op::gemp, be::cvsip,
                 void(Block0&, T, Block1 const&, Block2 const&, T)>
{
  typedef cvsip::op_traits<T> traits;

  static bool const ct_valid = 
    traits::valid &&
    is_same<T, typename Block0::value_type>::value &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block0, dda::inout>::ct_cost == 0 &&
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block0&, T, Block1 const&, Block2 const&, T)
  { return true;}
  static void exec(Block0& c, T alpha,
                   Block1 const& a, Block2 const& b, T beta)
  {
    namespace cc = cvsip;
    dda::Data<Block0, dda::inout> c_data(c);
    dda::Data<Block1, dda::in> a_data(a);
    dda::Data<Block2, dda::in> b_data(b);

    cc::View<2, T> cview(c_data.ptr(), 0,
			 c_data.stride(0), c.size(2, 0),
			 c_data.stride(1), c.size(2, 1));
    cc::const_View<2, T> aview(a_data.ptr(), 0,
			       a_data.stride(0), a.size(2, 0),
			       a_data.stride(1), a.size(2, 1));
    cc::const_View<2, T> bview(b_data.ptr(), 0,
			       b_data.stride(0), b.size(2, 0),
			       b_data.stride(1), b.size(2, 1));
    traits::gemp(cview.ptr(), alpha, aview.ptr(), bview.ptr(), beta);
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
