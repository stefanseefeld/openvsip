/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/fns_userelt.hpp
    @author  Stefan Seefeld
    @date    2005-07-27
    @brief   VSIPL++ Library: [math.fns.userelt]

    This file defines an extension mechanism for user-specified
    functions applied in expression templates.
*/

#ifndef VSIP_CORE_FNS_USERELT_HPP
#define VSIP_CORE_FNS_USERELT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/expr/binary_block.hpp>
#include <vsip/core/expr/ternary_block.hpp>
#include <functional>

namespace vsip
{
namespace impl
{

/// Unary function adapter for unary expression blocks.
/// The Operator template parameter of the expression block
/// template takes a single type argument, while user extension
/// functions take two. The Unary_function template provides
/// an adapter such that the Unary_function_return_type template
/// will do the right thing when given a 
/// 'Unary_function<Result, Function>::Type'.
template <typename R, typename F> 
struct Unary_function
{
  template <typename A>
  struct Type : std::unary_function<A, R>
  {
    static char const* name() { return "unary"; }

    Type(F f) : function_(f) {}

    R operator()(A a) const { return function_(a);}

    F function_;
  };
};

/// Partial specialization for Unary_function, taking a std::unary_function 
/// argument.
template <typename F>
struct Unary_function<typename F::result_type, F>
{
  template <typename Dummy>
  struct Type : std::unary_function<typename F::argument_type,
				    typename F::result_type>
  {
    static char const* name() { return "unary"; }

    Type(F f) : function_(f) {}

    typename F::result_type operator()(typename F::argument_type a) const 
    { return function_(a);}

    F function_;
  };
};

/// Partial specialization for Unary_function, taking a function pointer 
/// argument.
template <typename A, typename R>
struct Unary_function<R, R (*)(A)>
{
  template <typename Dummy>
  struct Type : std::pointer_to_unary_function<A, R>
  {
    static char const* name() { return "unary"; }

    Type(R (*f)(A)) : std::pointer_to_unary_function<A, R>(f) {}
  };
};

template <typename R, typename F> 
struct Binary_function
{
  template <typename A1, typename A2>
  struct Type : std::binary_function<A1, A2, R>
  {
    static char const* name() { return "binary"; }

    Type(F f) : function_(f) {}

    R operator()(A1 a1, A2 a2) const { return function_(a1, a2);}

    F function_;
  };
};

/// Partial specialization for Binary_function, taking a std::binary_function 
/// argument.
template <typename F>
struct Binary_function<typename F::result_type, F>
{
  template <typename A1, typename A2>
  struct Type : std::binary_function<typename F::first_argument_type,
                                     typename F::second_argument_type,
                                     typename F::result_type>
  {
    static char const* name() { return "binary"; }

    Type(F f) : function_(f) {}

    typename F::result_type operator()(typename F::first_argument_type a1,
				       typename F::second_argument_type a2) const
    { return function_(a1, a2);}

    F function_;
  };
};

/// Partial specialization for Binary_function, taking a function pointer 
/// argument.
template <typename A1, typename A2, typename R>
struct Binary_function<R, R (*)(A1, A2)> 
{
  template <typename Dummy1, typename Dummy2>
  struct Type : std::pointer_to_binary_function<A1, A2, R>
  {
    static char const* name() { return "binary"; }

    Type(R (*f)(A1, A2)) : std::pointer_to_binary_function<A1, A2, R>(f) {}
  };
};

template <typename R, typename F> 
struct Ternary_function
{
  template <typename A1, typename A2, typename A3>
  struct Type
  {
    typedef A1 first_argument_type;
    typedef A2 second_argument_type;
    typedef A3 third_argument_type;
    typedef R result_type;

    static char const* name() { return "ternary"; }

    Type(F f) : function_(f) {}

    R operator()(A1 a1, A2 a2, A3 a3) const { return function_(a1, a2, a3);}

    F function_;
  };
};

/// Partial specialization for Ternary_function, taking a function pointer 
/// argument.
template <typename A1, typename A2, typename A3, typename R>
struct Ternary_function<R, R (*)(A1, A2, A3)> 
{
  template <typename Dummy1, typename Dummy2, typename Dummy3>
  struct Type
  {
    typedef A1 first_argument_type;
    typedef A2 second_argument_type;
    typedef A3 third_argument_type;
    typedef R result_type;

    static char const* name() { return "ternary"; }

    Type(R (*f)(A1, A2, A3)) : function_(f) {}

    R operator()(A1 a1, A2 a2, A3 a3) const { return function_(a1, a2, a3);}

    R (*function_)(A1, A2, A3);
  };
};

/// These classes ({Unary,Binary,Ternary}_userelt_return_type)
/// determine the return types for the unary, binary, and ternary
/// functions, respectively.
///
/// They are necessary for Intel C++ (9.0), which has difficulty with the
///   'impl::Unary_function<R, F>::template Type'
/// member class template when used directly in the return types for
/// those functions.

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename T, typename Block>
struct Unary_userelt_return_type
{
  typedef
    typename impl::Unary_operator_return_type<
      View, T, Block, impl::Unary_function<R, F>::template Type>::view_type
    type;
};

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename LBlock,
	  typename RBlock>
struct Binary_userelt_traits
  : Binary_operator_traits<Binary_function<R, F>::template Type,
   			   View, LBlock, View, RBlock>
{
  typedef Binary_operator_traits<Binary_function<R, F>::template Type,
				 View, LBlock, View, RBlock> base_type;
  
  static typename base_type::type 
  create(F const &f, LBlock const &lblock, RBlock const &rblock)
  {
    typename base_type::operator_type op(f);
    typename base_type::block_type block(op, lblock, rblock);
    return typename base_type::type(block);
  }
};

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename Type1, typename Block1,
	  typename Type2, typename Block2,
	  typename Type3, typename Block3>
struct Ternary_userelt_return_type
{
  typedef
    typename impl::Ternary_func_return_type<
      View, Type1, Block1,
      View, Type2, Block2,
      View, Type3, Block3,
      impl::Ternary_function<R, F>::template Type>::view_type
    type;
};

} // namespace vsip::impl

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename T, typename Block>
typename impl::Unary_userelt_return_type<R, F, View, T, Block>::type
unary(F f, View<T, Block> v)
{
  typedef typename impl::Unary_function<R, F>::template Type<T> Function;
  typedef impl::Unary_operator_return_type<View, T, Block,
    impl::Unary_function<R, F>::template Type> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  return view_type(block_type(Function(f), v.block()));
}

template <typename R,
	  template <typename, typename> class View,
	  typename T, typename Block>
typename 
impl::Unary_operator_return_type<
  View, T, Block, impl::Unary_function<R, R(*)(T)>::template Type>::view_type
unary(R(*f)(T), View<T, Block> v)
{
  typedef typename impl::Unary_function<R, R(*)(T)>::template Type<T> Function;
  typedef impl::Unary_operator_return_type<View, T, Block,
    impl::Unary_function<R, R(*)(T)>::template Type> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  return view_type(block_type(Function(f), v.block()));
}

template <typename F,
	  template <typename, typename> class View,
	  typename T, typename Block>
typename 
impl::Unary_operator_return_type<
  View, T, Block, impl::Unary_function<typename F::result_type, F>
    ::template Type>::view_type
unary(F f, View<T, Block> v)
{
  typedef impl::Unary_operator_return_type<View, T, Block,
    impl::Unary_function<typename F::result_type, F>::template Type> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  return view_type(block_type(std::ptr_fun(f), v.block()));
}

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename LType, typename LBlock,
	  typename RType, typename RBlock>
typename impl::Binary_userelt_traits<R, F, View, LBlock, RBlock>::type
binary(F f, View<LType, LBlock> v1, View<RType, RBlock> v2)
{
  typedef impl::Binary_userelt_traits<R, F, View, LBlock, RBlock> traits;
  return traits::create(f, v1.block(), v2.block());
}

template <typename R,
	  template <typename, typename> class View,
	  typename LType, typename LBlock,
	  typename RType, typename RBlock>
typename impl::Binary_userelt_traits<
  R, R(*)(LType, RType), View, LBlock, RBlock>::type
binary(R(*f)(LType, RType), View<LType, LBlock> v1, View<RType, RBlock> v2)
{
  typedef impl::Binary_userelt_traits<R, R(*)(LType, RType), View, LBlock, RBlock> traits;
  return traits::create(f, v1.block(), v2.block());
}

template <typename F,
	  template <typename, typename> class View,
	  typename LType, typename LBlock,
	  typename RType, typename RBlock>
typename impl::Binary_operator_traits<
  impl::Binary_function<typename F::result_type, F>::template Type,
  View, LBlock, View, RBlock>::type
binary(F f, View<LType, LBlock> v1, View<RType, RBlock> v2)
{
  typedef impl::Binary_userelt_traits<typename F::result_type, F,
				      View, LBlock, RBlock> traits;
  return traits::create(f, v1.block(), v2.block());
}
template <typename R, typename F,
	  template <typename, typename> class View,
	  typename Type1, typename Block1,
	  typename Type2, typename Block2,
	  typename Type3, typename Block3>
typename impl::Ternary_userelt_return_type<
  R, F, View, Type1, Block1, Type2, Block2, Type3, Block3>::type
ternary(F f,
	View<Type1, Block1> v1,
	View<Type2, Block2> v2,
	View<Type3, Block3> v3)
{
  typedef typename 
    impl::Ternary_function<R, F>::template Type<Type1, Type2, Type3> Function;
  typedef impl::Ternary_func_return_type<
    View, Type1, Block1,
    View, Type2, Block2,
    View, Type3, Block3,
    impl::Ternary_function<R, F>::template Type> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  return view_type(block_type(Function(f), v1.block(), v2.block(), v3.block()));
}

template <typename R,
	  template <typename, typename> class View,
	  typename Type1, typename Block1,
	  typename Type2, typename Block2,
	  typename Type3, typename Block3>
typename 
impl::Ternary_func_return_type<
  View, Type1, Block1,
  View, Type2, Block2,
  View, Type3, Block3,
  impl::Ternary_function<R, R(*)(Type1, Type2, Type3)>
    ::template Type>::view_type
ternary(R(*f)(Type1, Type2, Type3),
	View<Type1, Block1> v1,
	View<Type2, Block2> v2,
	View<Type3, Block3> v3)
{
  typedef typename impl::Ternary_function<R, R(*)(Type1, Type2, Type3)>
    ::template Type<Type1, Type2, Type3> Function;
  typedef impl::Ternary_func_return_type<
  View, Type1, Block1,
  View, Type2, Block2,
  View, Type3, Block3,
  impl::Ternary_function<R, R(*)(Type1, Type2, Type3)>
    ::template Type> type_trait;
  typedef typename type_trait::block_type block_type;
  typedef typename type_trait::view_type view_type;

  return view_type(block_type(Function(f), v1.block(), v2.block(), v3.block()));
}

}

#endif
