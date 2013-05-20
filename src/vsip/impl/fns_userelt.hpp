//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_fns_userelt_hpp_
#define vsip_impl_fns_userelt_hpp_

#include <ovxx/view/operators.hpp>
#include <functional>

namespace ovxx
{
namespace userelt
{
/// Unary function adapter for unary expression blocks.
/// The Operator template parameter of the expression block
/// template takes a single type argument, while user extension
/// functions take two. The unary template provides
/// an adapter such that the operators::unary template
/// will do the right thing when given a 
/// 'unary<Result, Function>::Type'.
template <typename R, typename F> 
struct unary_traits
{
  template <typename A>
  struct Type : std::unary_function<A, R>
  {
    Type(F f) : function_(f) {}
    R operator()(A a) const { return function_(a);}
    F function_;
  };
};

/// Partial specialization, taking a std::unary_function argument.
template <typename F>
struct unary_traits<typename F::result_type, F>
{
  template <typename Dummy>
  struct Type : std::unary_function<typename F::argument_type,
				    typename F::result_type>
  {
    Type(F f) : function_(f) {}
    typename F::result_type operator()(typename F::argument_type a) const 
    { return function_(a);}
    F function_;
  };
};

/// Partial specialization, taking a function pointer argument.
template <typename A, typename R>
struct unary_traits<R, R (*)(A)>
{
  template <typename Dummy>
  struct Type : std::pointer_to_unary_function<A, R>
  {
    Type(R (*f)(A)) : std::pointer_to_unary_function<A, R>(f) {}
  };
};

template <typename R, typename F> 
struct binary_traits
{
  template <typename A1, typename A2>
  struct Type : std::binary_function<A1, A2, R>
  {
    Type(F f) : function_(f) {}
    R operator()(A1 a1, A2 a2) const { return function_(a1, a2);}
    F function_;
  };
};

template <typename F>
struct binary_traits<typename F::result_type, F>
{
  template <typename A1, typename A2>
  struct Type : std::binary_function<typename F::first_argument_type,
                                     typename F::second_argument_type,
                                     typename F::result_type>
  {
    Type(F f) : function_(f) {}
    typename F::result_type operator()(typename F::first_argument_type a1,
				       typename F::second_argument_type a2) const
    { return function_(a1, a2);}
    F function_;
  };
};

template <typename A1, typename A2, typename R>
struct binary_traits<R, R (*)(A1, A2)> 
{
  template <typename Dummy1, typename Dummy2>
  struct Type : std::pointer_to_binary_function<A1, A2, R>
  {
    Type(R (*f)(A1, A2)) : std::pointer_to_binary_function<A1, A2, R>(f) {}
  };
};

template <typename R, typename F> 
struct ternary_traits
{
  template <typename A1, typename A2, typename A3>
  struct Type
  {
    typedef A1 first_argument_type;
    typedef A2 second_argument_type;
    typedef A3 third_argument_type;
    typedef R result_type;

    Type(F f) : function_(f) {}
    R operator()(A1 a1, A2 a2, A3 a3) const { return function_(a1, a2, a3);}
    F function_;
  };
};

template <typename A1, typename A2, typename A3, typename R>
struct ternary_traits<R, R (*)(A1, A2, A3)> 
{
  template <typename Dummy1, typename Dummy2, typename Dummy3>
  struct Type
  {
    typedef A1 first_argument_type;
    typedef A2 second_argument_type;
    typedef A3 third_argument_type;
    typedef R result_type;

    Type(R (*f)(A1, A2, A3)) : function_(f) {}
    R operator()(A1 a1, A2 a2, A3 a3) const { return function_(a1, a2, a3);}
    R (*function_)(A1, A2, A3);
  };
};


/// These classes determine the return types for the unary, binary, and ternary
/// functions, respectively.
///
/// They are necessary for Intel C++ (9.0), which has difficulty with the
///   'unary_traits<R, F>::template Type'
/// member class template when used directly in the return types for
/// those functions.

template <typename R, typename F,
	  template <typename, typename> class V,
	  typename T, typename B>
struct unary
{
  typedef typename 
  operators::unary<unary_traits<R, F>::template Type, V, T, B>::view_type
    type;
};

template <typename R, typename F,
	  template <typename, typename> class V,
	  typename B1,
	  typename B2>
struct binary : operators::binary<binary_traits<R, F>::template Type,
				  V, B1, V, B2>
{
  typedef operators::binary<binary_traits<R, F>::template Type,
			    V, B1, V, B2> base_type;
  
  static typename base_type::type 
  create(F const &f, B1 const &b1, B2 const &b2)
  {
    typename base_type::operator_type op(f);
    typename base_type::block_type block(op, b1, b2);
    return typename base_type::type(block);
  }
};

template <typename R, typename F,
	  template <typename, typename> class V,
	  typename T1, typename B1,
	  typename T2, typename B2,
	  typename T3, typename B3>
struct ternary
{
  typedef typename functors::ternary<ternary_traits<R, F>::template Type,
				     V, T1, B1,
				     V, T2, B2,
				     V, T3, B3>::view_type
    type;
};

} // namespace ovxx::userelt
} // namespace ovxx

namespace vsip
{
template <typename R, typename F,
	  template <typename, typename> class View,
	  typename T, typename Block>
typename ovxx::userelt::unary<R, F, View, T, Block>::type
unary(F f, View<T, Block> v)
{
  using namespace ovxx;
  typedef typename userelt::unary_traits<R, F>::template Type<T> Function;
  typedef operators::unary<userelt::unary_traits<R, F>::template Type,
		View, T, Block> type_traits;
  typedef typename type_traits::block_type block_type;
  typedef typename type_traits::view_type view_type;
  return view_type(block_type(Function(f), v.block()));
}

template <typename R,
	  template <typename, typename> class View,
	  typename T, typename Block>
typename ovxx::operators::unary<
  ovxx::userelt::unary_traits<R, R(*)(T)>::template Type,
  View, T, Block>::view_type
unary(R(*f)(T), View<T, Block> v)
{
  using namespace ovxx;
  typedef typename userelt::unary_traits<R, R(*)(T)>::template Type<T> Function;
  typedef operators::unary<userelt::unary_traits<R, R(*)(T)>::template Type,
		      View, T, Block> type_traits;
  typedef typename type_traits::block_type block_type;
  typedef typename type_traits::view_type view_type;
  return view_type(block_type(Function(f), v.block()));
}

template <typename F,
	  template <typename, typename> class View,
	  typename T, typename Block>
typename ovxx::operators::unary<
  ovxx::userelt::unary_traits<typename F::result_type, F>::template Type,
  View, T, Block>::view_type
unary(F f, View<T, Block> v)
{
  using namespace ovxx;
  typedef operators::unary<
    userelt::unary_traits<typename F::result_type, F>::template Type,
    View, T, Block> type_traits;
  typedef typename type_traits::block_type block_type;
  typedef typename type_traits::view_type view_type;
  return view_type(block_type(std::ptr_fun(f), v.block()));
}

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename LType, typename LBlock,
	  typename RType, typename RBlock>
typename ovxx::userelt::binary<R, F, View, LBlock, RBlock>::type
binary(F f, View<LType, LBlock> v1, View<RType, RBlock> v2)
{
  typedef ovxx::userelt::binary<R, F, View, LBlock, RBlock> traits;
  return traits::create(f, v1.block(), v2.block());
}

template <typename R,
	  template <typename, typename> class View,
	  typename LType, typename LBlock,
	  typename RType, typename RBlock>
typename ovxx::userelt::binary<
  R, R(*)(LType, RType), View, LBlock, RBlock>::type
binary(R(*f)(LType, RType), View<LType, LBlock> v1, View<RType, RBlock> v2)
{
  typedef ovxx::userelt::binary<R, R(*)(LType, RType), View, LBlock, RBlock> traits;
  return traits::create(f, v1.block(), v2.block());
}

template <typename F,
	  template <typename, typename> class View,
	  typename LType, typename LBlock,
	  typename RType, typename RBlock>
typename ovxx::operators::binary<
  ovxx::userelt::binary_traits<typename F::result_type, F>::template Type,
  View, LBlock, View, RBlock>::type
binary(F f, View<LType, LBlock> v1, View<RType, RBlock> v2)
{
  typedef ovxx::userelt::binary<typename F::result_type, F,
		       View, LBlock, RBlock> traits;
  return traits::create(f, v1.block(), v2.block());
}

template <typename R, typename F,
	  template <typename, typename> class View,
	  typename Type1, typename Block1,
	  typename Type2, typename Block2,
	  typename Type3, typename Block3>
typename ovxx::userelt::ternary<
  R, F, View, Type1, Block1, Type2, Block2, Type3, Block3>::type
ternary(F f,
	View<Type1, Block1> v1,
	View<Type2, Block2> v2,
	View<Type3, Block3> v3)
{
  using namespace ovxx;
  typedef typename 
    userelt::ternary_traits<R, F>::template Type<Type1, Type2, Type3> Function;
  typedef functors::ternary<userelt::ternary_traits<R, F>::template Type,
			    View, Type1, Block1,
			    View, Type2, Block2,
			    View, Type3, Block3> type_traits;
  typedef typename type_traits::block_type block_type;
  typedef typename type_traits::view_type view_type;

  return view_type(block_type(Function(f), v1.block(), v2.block(), v3.block()));
}

template <typename R,
	  template <typename, typename> class View,
	  typename Type1, typename Block1,
	  typename Type2, typename Block2,
	  typename Type3, typename Block3>
typename 
ovxx::functors::ternary<
  ovxx::userelt::ternary_traits<R, R(*)(Type1, Type2, Type3)>::template Type,
  View, Type1, Block1,
  View, Type2, Block2,
  View, Type3, Block3>
    ::view_type
ternary(R(*f)(Type1, Type2, Type3),
	View<Type1, Block1> v1,
	View<Type2, Block2> v2,
	View<Type3, Block3> v3)
{
  using namespace ovxx;
  typedef typename userelt::ternary_traits<R, R(*)(Type1, Type2, Type3)>
    ::template Type<Type1, Type2, Type3> Function;
  typedef functors::ternary<
    userelt::ternary_traits<R, R(*)(Type1, Type2, Type3)>::template Type,
    View, Type1, Block1,
    View, Type2, Block2,
    View, Type3, Block3> type_traits;
  typedef typename type_traits::block_type block_type;
  typedef typename type_traits::view_type view_type;
  return view_type(block_type(Function(f), v1.block(), v2.block(), v3.block()));
}

} // namespace vsip

#endif
