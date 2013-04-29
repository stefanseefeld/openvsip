/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/functor.hpp
    @author  Stefan Seefeld
    @date    2005-04-21
    @brief   VSIPL++ Library: Functor helper templates.

    This file declares templates to be used with function expression templates.
*/

#ifndef VSIP_CORE_EXPR_FUNCTOR_HPP
#define VSIP_CORE_EXPR_FUNCTOR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/expr/binary_block.hpp>
#include <vsip/core/expr/ternary_block.hpp>
#include <vsip/core/expr/binary_operators.hpp>

namespace vsip
{
namespace impl
{

/// Unary_func_return_type encapsulates the inference
/// of a (block) function's return type from its arguments.
template <template <typename, typename> class View,
 	  typename Type,
	  typename Block,
          template <typename> class Functor>
struct Unary_func_return_type
{
  typedef typename Functor<Type>::result_type value_type;

  typedef const expr::Unary<Functor, Block, true> block_type;

  typedef typename ViewConversion<View, 
				  value_type, 
				  block_type>::const_view_type view_type;
};

template <template <typename, typename> class View1,
 	  typename Type1, typename Block1,
          template <typename, typename> class View2,
 	  typename Type2, typename Block2,
          template <typename, typename> class Functor>
struct Binary_func_return_type
{
  typedef typename Functor<Type1, Type2>::result_type value_type;

  typedef const expr::Binary<Functor, Block1, Block2, true> block_type;

  // The following projects the two view types onto their respective
  // const equivalent and promotes these to a return type.
  // Thus, Binary_operator_return_type(const_Matrix, Matrix) -> const_Matrix
  // but Binary_operator_return_type(Vector, Matrix) -> failure.
  typedef typename 
  View_promotion<typename ViewConversion<View1,
					value_type,
					block_type>::const_view_type,
		typename ViewConversion<View1,
					value_type,
					block_type>::const_view_type>::type 
					view_type;
};

template <template <typename, typename> class View1,
 	  typename Type1, typename Block1,
          template <typename, typename> class View2,
 	  typename Type2, typename Block2,
          template <typename, typename> class View3,
 	  typename Type3, typename Block3,
          template <typename, typename, typename> class Functor>
struct Ternary_func_return_type
{
  typedef typename Functor<Type1, Type2, Type3>::result_type value_type;

  typedef const expr::Ternary<Functor,
			      Block1,
			      Block2,
			      Block3,
			      true> block_type;

  // The following projects the two view types onto their respective
  // const equivalent and promotes these to a return type.
  // Thus, Binary_operator_return_type(const_Matrix, Matrix) -> const_Matrix
  // but Binary_operator_return_type(Vector, Matrix) -> failure.
  typedef typename 
  View_promotion<typename View_promotion<
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

template <template <typename> class Functor,
	  typename View>
struct Unary_func_expr;

template <template <typename> class Functor,
	  template <typename, typename> class View,
	  typename Type, typename Block>
struct Unary_func_expr<Functor, View<Type, Block> >
{
  typedef Unary_func_return_type<View, Type, Block, Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View<Type, Block> const& view)
  {
    return view_type(block_type(view.block()));
  }
};

template <template <typename, typename> class Functor,
	  typename View1, typename View2>
struct Binary_func_expr;

template <template <typename, typename> class Functor,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2>
struct Binary_func_expr<Functor, View1<Type1, Block1>, View2<Type2, Block2> >
{
  typedef Binary_func_return_type<View1, Type1, Block1,
				  View2, Type2, Block2,
				  Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View1<Type1, Block1> const& view1,
			  View2<Type2, Block2> const& view2)
  {
    return view_type(block_type(view1.block(), view2.block()));
  }
};

template <template <typename, typename> class Functor,
	  template <typename, typename> class View,
	  typename Type, typename Block,
	  typename S>
struct Binary_func_expr<Functor,
			View<Type, Block>,
			S>
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef Binary_func_return_type<View, Type, Block,
				  View, S, scalar_block_type const,
				  Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View<Type, Block> const& view, S scalar)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(view.block(), sblock));
  }
};

template <template <typename, typename> class Functor,
	  typename S,
	  template <typename, typename> class View,
	  typename Type, typename Block>
struct Binary_func_expr<Functor,
			S,
			View<Type, Block> >
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef Binary_func_return_type<View, S, scalar_block_type const,
                                  View, Type, Block,
                                  Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S scalar, View<Type, Block> const& view)
  {
    scalar_block_type sblock(scalar);
    return view_type(block_type(sblock, view.block()));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename View1, typename View2, typename View3>
struct Ternary_func_expr;

template <template <typename, typename, typename> class Functor,
	  template <typename, typename> class View1,
	  typename Type1, typename Block1,
	  template <typename, typename> class View2,
	  typename Type2, typename Block2,
	  template <typename, typename> class View3,
	  typename Type3, typename Block3>
struct Ternary_func_expr<Functor,
			 View1<Type1, Block1>,
			 View2<Type2, Block2>,
			 View3<Type3, Block3> >
{
  typedef Ternary_func_return_type<View1, Type1, Block1,
				   View2, Type2, Block2,
				   View3, Type3, Block3,
				   Functor> type_trait;

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
struct Ternary_func_expr<Functor,
			 View1<Type1, Block1>,
			 View2<Type2, Block2>,
			 S>
{
  static dimension_type const dim = View1<Type1, Block1>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef Ternary_func_return_type<View1, Type1, Block1,
				   View2, Type2, Block2,
				   View1, S, scalar_block_type const,
				   Functor> type_trait;

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
struct Ternary_func_expr<Functor,
			 View1<Type1, Block1>,
			 S,
			 View2<Type2, Block2> >
{
  static dimension_type const dim = View1<Type1, Block1>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef Ternary_func_return_type<View1, Type1, Block1,
				   View1, S, scalar_block_type const,
				   View2, Type2, Block2,
				   Functor> type_trait;

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
struct Ternary_func_expr<Functor,
			 S,
			 View1<Type1, Block1>,
			 View2<Type2, Block2> >
{
  static dimension_type const dim = View1<Type1, Block1>::dim;
  typedef expr::Scalar<dim, S> scalar_block_type;
  typedef Ternary_func_return_type<View1, S, scalar_block_type const,
				   View1, Type1, Block1,
				   View2, Type2, Block2,
				   Functor> type_trait;

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
	  template <typename, typename> class View,
	  typename Type, typename Block,
	  typename S1,
	  typename S2>
struct Ternary_func_expr<Functor,
			 View<Type, Block>,
			 S1,
			 S2>
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S1> scalar_block1_type;
  typedef expr::Scalar<dim, S2> scalar_block2_type;
  typedef Ternary_func_return_type<View, Type, Block,
				   View, S1, scalar_block1_type const,
				   View, S2, scalar_block2_type const,
				   Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(View<Type, Block> const& view,
			  S1 scalar1,
			  S2 scalar2)
  {
    scalar_block1_type sblock1(scalar1);
    scalar_block2_type sblock2(scalar2);
    return view_type(block_type(view.block(),
				sblock1,
				sblock2));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename S1,
	  template <typename, typename> class View,
	  typename Type, typename Block,
	  typename S2>
struct Ternary_func_expr<Functor,
			 S1,
			 View<Type, Block>,
			 S2>
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S1> scalar_block1_type;
  typedef expr::Scalar<dim, S2> scalar_block2_type;
  typedef Ternary_func_return_type<View, S1, scalar_block1_type const,
				   View, Type, Block,
				   View, S2, scalar_block2_type const,
				   Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S1 scalar1,
			  View<Type, Block> const& view,
			  S2 scalar2)
  {
    scalar_block1_type sblock1(scalar1);
    scalar_block2_type sblock2(scalar2);
    return view_type(block_type(sblock1,
				view.block(),
				sblock2));
  }
};

template <template <typename, typename, typename> class Functor,
	  typename S1,
	  typename S2,
	  template <typename, typename> class View,
	  typename Type, typename Block>
struct Ternary_func_expr<Functor,
			 S1,
			 S2,
			 View<Type, Block> >
{
  static dimension_type const dim = View<Type, Block>::dim;
  typedef expr::Scalar<dim, S1> scalar_block1_type;
  typedef expr::Scalar<dim, S2> scalar_block2_type;
  typedef Ternary_func_return_type<View, S1, scalar_block1_type const,
				   View, S2, scalar_block2_type const,
				   View, Type, Block,
				   Functor> type_trait;

  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  static view_type create(S1 scalar1,
			  S2 scalar2,
			  View<Type, Block> const& view)
  {
    scalar_block1_type sblock1(scalar1);
    scalar_block2_type sblock2(scalar2);
    return view_type(block_type(sblock1,
				sblock2,
				view.block()));
  }
};

template <template <typename> class Func_scalar,
	  typename T>
struct Unary_func_view
{
  typedef Unary_func_expr<Func_scalar, T> expr;
  typedef typename expr::view_type result_type;
  static result_type apply(T const& value)
  { return expr::create(value);}
};

template <template <typename, typename> class Func_scalar,
	  typename T1, typename T2>
struct Binary_func_view
{
  typedef Binary_func_expr<Func_scalar, T1, T2> expr;
  typedef typename expr::view_type result_type;
  static result_type apply(T1 const& value1, T2 const& value2)
  { return expr::create(value1, value2);}
};

template <template <typename, typename, typename> class Func_scalar,
	  typename T1, typename T2, typename T3>
struct Ternary_func_view
{
  typedef Ternary_func_expr<Func_scalar, T1, T2, T3> expr;
  typedef typename expr::view_type result_type;
  static result_type apply(T1 const& value1, T2 const& value2,
			   T3 const& value3)
  { return expr::create(value1, value2, value3);}
};

} // namespace vsip::impl
} // namespace vsip

#endif
