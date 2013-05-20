//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_operators_hpp_
#define ovxx_view_operators_hpp_

#include <ovxx/expr/unary.hpp>
#include <ovxx/expr/binary.hpp>
#include <ovxx/detail/lazy_enable_if.hpp>
#include <ovxx/view.hpp>
#include <vsip/impl/promotion.hpp>
#include <ovxx/view/fns_elementwise.hpp>

namespace ovxx
{
namespace operators
{
using functors::view_promotion;

template <template <typename> class O,
	  template <typename, typename> class V, typename T, typename B,
	  typename Tag = typename is_view_type<V<T, B> >::type>
struct unary
{
  typedef T value_type;
  typedef expr::Unary<O, B, true> const block_type;
  typedef typename ViewConversion<V, value_type, block_type>::const_view_type view_type;
};

template <template <typename, typename> class O,
	  template <typename, typename> class LView, typename LBlock,
 	  template <typename, typename> class RView, typename RBlock,
	  typename Enable = void>
struct binary
{
  typedef O<typename LBlock::value_type, typename RBlock::value_type> operator_type;
  typedef const expr::Binary<O, LBlock, RBlock, true> block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  view_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(LBlock const &lblock, RBlock const &rblock)
  {
    block_type block(lblock, rblock);
    return type(block);
  }
};

/// Fuse A * B + C into a Ternary expression block.
template <template <typename, typename> class LView,
	  typename ABlock, typename BBlock,
	  template <typename, typename> class RView,
	  typename CBlock>
struct binary<expr::op::Add,
  LView, expr::Binary<expr::op::Mult, ABlock, BBlock, true> const,
  RView, CBlock>
{
  typedef expr::Binary<expr::op::Mult, ABlock, BBlock, true> l_block_type;
  typedef expr::Ternary<expr::op::Ma, ABlock, BBlock, CBlock, true> const block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  view_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(l_block_type const &lblock, CBlock const &cblock)
  {
    block_type block(lblock.arg1(), lblock.arg2(), cblock);
    return type(block);
  }
};

/// Fuse A + B * C into a Ternary expression block.
template <template <typename, typename> class LView,
	  typename ABlock,
	  template <typename, typename> class RView,
	  typename BBlock, typename CBlock>
struct binary<expr::op::Add,
  LView, ABlock,
  RView, expr::Binary<expr::op::Mult, BBlock, CBlock, true> const>
{
  typedef expr::Binary<expr::op::Mult, BBlock, CBlock, true> r_block_type;
  typedef expr::Ternary<expr::op::Ma, BBlock, CBlock, ABlock, true> const block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  view_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(ABlock const &ablock, r_block_type const &rblock)
  {
    block_type block(rblock.arg1(), rblock.arg2(), ablock);
    return type(block);
  }
};

/// Fuse A * B + C * D into a Ternary expression block.
/// (This maps to Ternary<A, B, Binary<B, C> >.)
template <template <typename, typename> class LView,
	  typename ABlock, typename BBlock,
	  template <typename, typename> class RView,
	  typename CBlock, typename DBlock>
struct binary<expr::op::Add,
  LView, expr::Binary<expr::op::Mult, ABlock, BBlock, true> const,
  RView, expr::Binary<expr::op::Mult, CBlock, DBlock, true> const>
{
  typedef expr::Binary<expr::op::Mult, ABlock, BBlock, true> l_block_type;
  typedef expr::Binary<expr::op::Mult, CBlock, DBlock, true> r_block_type;
  typedef expr::Ternary<expr::op::Ma, ABlock, BBlock, r_block_type const, true> const block_type;
  typedef typename block_type::value_type value_type;

  typedef typename 
  view_promotion<typename ViewConversion<LView,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<RView,
					 value_type,
					 block_type>::const_view_type>::type 
    type; // 'type' is the expected name lazy_enable_if relies on.

  static type create(l_block_type const &lblock, r_block_type const &rblock)
  {
    block_type block(lblock.arg1(), lblock.arg2(), rblock);
    return type(block);
  }
};

} // namespace ovxx::operators

// Operators: They are defined in ovxx::, but are found by ADL

template <template <typename, typename> class V, typename T, typename B>
inline
typename operators::unary<expr::op::Plus, V, T, B>::view_type
operator+(V<T, B> const &view)
{
  typedef operators::unary<expr::op::Plus, V, T, B> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;
  return view_type(block_type(view.block()));
}

template <template <typename, typename> class V, typename T, typename B>
inline
typename operators::unary<expr::op::Minus, V, T, B>::view_type
operator-(V<T, B> const &view)
{
  typedef operators::unary<expr::op::Minus, V, T, B> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;
  return view_type(block_type(view.block()));
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Add,
    V, B, V, expr::Scalar<V<T, B>::dim, S> const> >::type
operator+(V<T, B> const &lhs, S rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Add, V, B, V, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Add,
    V, expr::Scalar<V<T, B>::dim, S> const, V, B> >::type
operator+(S lhs, V<T, B> const &rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Add, V, scalar_block_type, V, B> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class V1,
 	  typename T1, typename B1,
 	  template <typename, typename> class V2,
 	  typename T2, typename B2>
inline
typename detail::lazy_enable_if_c<
  is_view_type<V1<T1, B1> >::value &&
  is_view_type<V2<T2, B2> >::value,
  operators::binary<expr::op::Add, V1, B1, V2, B2> >::type
operator+(V1<T1, B1> const &lhs, V2<T2, B2> const &rhs)
{
  typedef operators::binary<expr::op::Add, V1, B1, V2, B2> traits;
  return traits::create(lhs.block(), rhs.block());
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Sub, V, B, V, expr::Scalar<V<T, B>::dim, S> const> >::type
operator-(V<T, B> const &lhs, S rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Sub, V, B, V, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Sub,
    V, expr::Scalar<V<T, B>::dim, S> const, V, B> >::type
operator-(S lhs, V<T, B> const &rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Sub, V, scalar_block_type, V, B> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class V1,
 	  typename T1, typename B1,
 	  template <typename, typename> class V2,
 	  typename T2, typename B2>
inline
typename detail::lazy_enable_if_c<
  is_view_type<V1<T1, B1> >::value &&
  is_view_type<V2<T2, B2> >::value,
  operators::binary<expr::op::Sub, V1, B1, V2, B2> >::type
operator-(V1<T1, B1> const &lhs, V2<T2, B2> const &rhs)
{
  typedef operators::binary<expr::op::Sub, V1, B1, V2, B2> traits;
  return traits::create(lhs.block(), rhs.block());
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Mult,
    V, B, V, expr::Scalar<V<T, B>::dim, S> const> >::type
operator*(V<T, B> const &lhs, S rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Mult, V, B, V, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Mult,
    V, expr::Scalar<V<T, B>::dim, S> const, V, B> >::type
operator*(S lhs, V<T, B> const &rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Mult, V, scalar_block_type, V, B> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class V1,
 	  typename T1, typename B1,
 	  template <typename, typename> class V2,
 	  typename T2, typename B2>
inline
typename detail::lazy_enable_if_c<
  is_view_type<V1<T1, B1> >::value &&
  is_view_type<V2<T2, B2> >::value,
  operators::binary<expr::op::Mult, V1, B1, V2, B2> >::type
operator*(V1<T1, B1> const &lhs, V2<T2, B2> const &rhs)
{
  typedef operators::binary<expr::op::Mult, V1, B1, V2, B2> traits;
  return traits::create(lhs.block(), rhs.block());
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Div,
    V, B, V, expr::Scalar<V<T, B>::dim, S> const> >::type
operator/(V<T, B> const &lhs, S rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Div, V, B, V, scalar_block_type> traits;
  return traits::create(lhs.block(), scalar_block_type(rhs));
}

template <template <typename, typename> class V, typename T, typename B, typename S>
inline
typename detail::lazy_enable_if<is_view_type<V<T, B> >,
  operators::binary<expr::op::Div,
  V, expr::Scalar<V<T, B>::dim, S> const, V, B> >::type
operator/(S lhs, V<T, B> const &rhs)
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> const scalar_block_type;
  typedef operators::binary<expr::op::Div, V, scalar_block_type, V, B> traits;
  return traits::create(scalar_block_type(lhs), rhs.block());
}

template <template <typename, typename> class V1,
 	  typename T1, typename B1,
 	  template <typename, typename> class V2,
 	  typename T2, typename B2>
inline
typename detail::lazy_enable_if_c<
  is_view_type<V1<T1, B1> >::value &&
  is_view_type<V2<T2, B2> >::value,
  operators::binary<expr::op::Div, V1, B1, V2, B2> >::type
operator/(V1<T1, B1> const &lhs, V2<T2, B2> const &rhs)
{
  typedef operators::binary<expr::op::Div, V1, B1, V2, B2> traits;
  return traits::create(lhs.block(), rhs.block());
}

} // namespace ovxx

#endif
