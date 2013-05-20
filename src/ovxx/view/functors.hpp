//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_functors_hpp_
#define ovxx_view_functors_hpp_

#include <ovxx/expr/unary.hpp>
#include <ovxx/expr/binary.hpp>
#include <ovxx/expr/ternary.hpp>
#include <vsip/impl/view_fwd.hpp>

namespace ovxx
{
namespace functors
{
template <typename LView, typename RView>
struct view_promotion {};

template <typename View>
struct view_promotion<View, View>
{
  typedef View type;
};

/// unary encapsulates the inference
/// of a (block) function's return type from its arguments.
template <template <typename> class Functor,
	  template <typename, typename> class V, typename T, typename B>
struct unary
{
  typedef typename Functor<T>::result_type value_type;
  typedef const ovxx::expr::Unary<Functor, B, true> block_type;
  typedef typename ViewConversion<V, 
				  value_type, 
				  block_type>::const_view_type view_type;
};

template <template <typename, typename> class Functor,
	  template <typename, typename> class View1,
 	  typename Type1, typename Block1,
          template <typename, typename> class View2,
 	  typename Type2, typename Block2>
struct binary
{
  typedef typename Functor<Type1, Type2>::result_type value_type;
  typedef const ovxx::expr::Binary<Functor, Block1, Block2, true> block_type;

  // The following projects the two view types onto their respective
  // const equivalent and promotes these to a return type.
  // Thus, Binary_operator_return_type(const_Matrix, Matrix) -> const_Matrix
  // but Binary_operator_return_type(Vector, Matrix) -> failure.
  typedef typename 
  view_promotion<typename ViewConversion<View1,
					 value_type,
					 block_type>::const_view_type,
		 typename ViewConversion<View1,
					 value_type,
					 block_type>::const_view_type>::type 
    view_type;
};

template <template <typename, typename, typename> class Functor,
          template <typename, typename> class View1,
 	  typename Type1, typename Block1,
          template <typename, typename> class View2,
 	  typename Type2, typename Block2,
          template <typename, typename> class View3,
 	  typename Type3, typename Block3>
struct ternary
{
  typedef typename Functor<Type1, Type2, Type3>::result_type value_type;
  typedef const ovxx::expr::Ternary<Functor,
				    Block1,
				    Block2,
				    Block3,
				    true> block_type;

  // The following projects the two view types onto their respective
  // const equivalent and promotes these to a return type.
  // Thus, Binary_operator_return_type(const_Matrix, Matrix) -> const_Matrix
  // but Binary_operator_return_type(Vector, Matrix) -> failure.
  typedef typename 
  view_promotion<
    typename view_promotion<
      typename ViewConversion<View1,
  			      value_type,
			      block_type>::const_view_type,
      typename ViewConversion<View2,
			      value_type,
			      block_type>::const_view_type>::type,
	typename ViewConversion<View3,
				value_type,
				block_type>::const_view_type>::type 
          view_type;
};

template <template <typename> class Functor, typename View>
struct unary_expr;

template <template <typename> class Functor,
	  template <typename, typename> class V, typename T, typename B>
struct unary_expr<Functor, V<T, B> >
{
  typedef unary<Functor, V, T, B> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(V<T, B> const &view)
  {
    return view_type(block_type(view.block()));
  }
};

template <template <typename, typename> class Functor,
	  typename View1, typename View2>
struct binary_expr;

template <template <typename, typename> class Functor,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2>
struct binary_expr<Functor, View1<Type1, Block1>, View2<Type2, Block2> >
{
  typedef binary<Functor,
		 View1, Type1, Block1,
		 View2, Type2, Block2> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View1<Type1, Block1> const& view1,
			  View2<Type2, Block2> const& view2)
  {
    return view_type(block_type(view1.block(), view2.block()));
  }
};

template <template <typename, typename> class Functor,
	  template <typename, typename> class V,
	  typename T, typename B,
	  typename S>
struct binary_expr<Functor, V<T, B>, S>
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef binary<Functor,
		 V, T, B,
		 V, S, scalar_block_type const> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(V<T, B> const &view, S scalar)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(view.block(), sblock));
  }
};

template <template <typename, typename> class Functor,
	  typename S,
	  template <typename, typename> class V,
	  typename T, typename B>
struct binary_expr<Functor, S, V<T, B> >
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef binary<Functor, V, S, scalar_block_type const, V, T, B> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S scalar, V<T, B> const &view)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(sblock, view.block()));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename View1, typename View2, typename View3>
struct ternary_expr;

template <template <typename, typename, typename> class Functor,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2,
	  template <typename, typename> class View3,
	  typename Type3, typename Block3>
struct ternary_expr<Functor,
		    View1<Type1, Block1>,
		    View2<Type2, Block2>,
		    View3<Type3, Block3> >
{
  typedef ternary<Functor,
		  View1, Type1, Block1,
		  View2, Type2, Block2,
		  View3, Type3, Block3> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View1<Type1, Block1> const& view1,
			  View2<Type2, Block2> const& view2,
			  View3<Type3, Block3> const& view3)
  {
    return view_type(block_type(view1.block(),
				view2.block(),
				view3.block()));
  }
};

template <template <typename, typename, typename> class Functor,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2,
	  typename S>
struct ternary_expr<Functor,
		    View1<Type1, Block1>,
		    View2<Type2, Block2>,
		    S>
{
  static dimension_type const dim = View1<Type1, Block1>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef ternary<Functor,
		  View1, Type1, Block1,
		  View2, Type2, Block2,
		  View1, S, scalar_block_type const> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View1<Type1, Block1> const& view1,
			  View2<Type2, Block2> const& view2,
			  S scalar)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(view1.block(),
				view2.block(),
				sblock));
  }
};

template <template <typename, typename, typename> class Functor,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  typename S,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2>
struct ternary_expr<Functor,
		    View1<Type1, Block1>,
		    S,
		    View2<Type2, Block2> >
{
  static dimension_type const dim = View1<Type1, Block1>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef ternary<Functor,
		  View1, Type1, Block1,
		  View1, S, scalar_block_type const,
		  View2, Type2, Block2> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View1<Type1, Block1> const& view1,
			  S scalar,
			  View2<Type2, Block2> const& view2)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(view1.block(),
				sblock,
				view2.block()));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename S,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2>
struct ternary_expr<Functor,
		    S,
		    View1<Type1, Block1>,
		    View2<Type2, Block2> >
{
  static dimension_type const dim = View1<Type1, Block1>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef ternary<Functor,
		  View1, S, scalar_block_type const,
		  View1, Type1, Block1,
		  View2, Type2, Block2> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S scalar,
			  View1<Type1, Block1> const& view1,
			  View2<Type2, Block2> const& view2)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(sblock,
				view1.block(),
				view2.block()));
  }
};

template <template <typename, typename, typename> class Functor,
	  template <typename, typename> class V, typename T, typename B,
	  typename S1,
	  typename S2>
struct ternary_expr<Functor, V<T, B>, S1, S2>
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S1> scalar_block1_type;
  typedef expr::Scalar<dim, S2> scalar_block2_type;
  typedef ternary<Functor,
		  V, T, B,
		  V, S1, scalar_block1_type const,
		  V, S2, scalar_block2_type const> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(V<T, B> const& view, S1 scalar1, S2 scalar2)
  {
    scalar_block1_type sblock1(scalar1);
    scalar_block2_type sblock2(scalar2);
    return view_type(block_type(view.block(), sblock1, sblock2));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename S1,
	  template <typename, typename> class V, typename T, typename B,
	  typename S2>
struct ternary_expr<Functor, S1, V<T, B>, S2>
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S1> scalar_block1_type;
  typedef expr::Scalar<dim, S2> scalar_block2_type;
  typedef ternary<Functor,
		  V, S1, scalar_block1_type const,
		  V, T, B,
		  V, S2, scalar_block2_type const> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S1 scalar1, V<T, B> const& view, S2 scalar2)
  {
    scalar_block1_type sblock1(scalar1);
    scalar_block2_type sblock2(scalar2);
    return view_type(block_type(sblock1, view.block(), sblock2));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename S1,
	  typename S2,
	  template <typename, typename> class V, typename T, typename B>
struct ternary_expr<Functor, S1, S2, V<T, B> >
{
  static dimension_type const dim = V<T, B>::dim;
  typedef expr::Scalar<dim, S1> scalar_block1_type;
  typedef expr::Scalar<dim, S2> scalar_block2_type;
  typedef ternary<Functor,
		  V, S1, scalar_block1_type const,
		  V, S2, scalar_block2_type const,
		  V, T, B> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S1 scalar1, S2 scalar2, V<T, B> const &view)
  {
    scalar_block1_type sblock1(scalar1);
    scalar_block2_type sblock2(scalar2);
    return view_type(block_type(sblock1, sblock2, view.block()));
  }
};

template <template <typename> class Func_scalar,
	  typename T>
struct unary_view
{
  typedef unary_expr<Func_scalar, T> expr;
  typedef typename expr::view_type result_type;
  static result_type apply(T const& value)
  { return expr::create(value);}
};

template <template <typename, typename> class Func_scalar,
	  typename T1, typename T2>
struct binary_view
{
  typedef binary_expr<Func_scalar, T1, T2> expr;
  typedef typename expr::view_type result_type;
  static result_type apply(T1 const& value1, T2 const& value2)
  { return expr::create(value1, value2);}
};

template <template <typename, typename, typename> class Func_scalar,
	  typename T1, typename T2, typename T3>
struct ternary_view
{
  typedef ternary_expr<Func_scalar, T1, T2, T3> expr;
  typedef typename expr::view_type result_type;
  static result_type apply(T1 const& value1, T2 const& value2, T3 const& value3)
  { return expr::create(value1, value2, value3);}
};

} // namespace ovxx::functors
} // namespace ovxx

#endif
