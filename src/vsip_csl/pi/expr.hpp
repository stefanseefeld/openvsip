/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression support

#ifndef vsip_csl_pi_expr_hpp_
#define vsip_csl_pi_expr_hpp_

#include <vsip_csl/pi/iterator.hpp>

namespace vsip_csl
{
namespace pi
{

/// A map allows to write iterator functions,
/// such as iterator lookup tables.
class Map
{
  typedef index_type (*F)(index_type);
public:
  template <typename I>
  Map(F f, I i) : func_(f), offset_(i + 0) {}
  index_type apply(index_type i) { return func_(i + offset_);}

private:
  F func_;
  stride_type offset_;
};

template <> 
struct is_iterator<Map> { static bool const value = true;};

template <>
struct is_expr<Map> { static bool const value = true;};

/// Scalar represents a numeric literal in an expression.
template <typename T>
class Scalar
{
public:
  typedef T result_type;

  Scalar(T t) : value_(t) {}
  T value() const { return value_;}

private:
  T value_;
};

/// Unary represents a unary operation in an expression.
template <template <typename> class O, typename A>
class Unary
{
  typedef O<typename A::result_type> operation_type;

public:
  typedef typename operation_type::result_type result_type;

  Unary(A a) : arg_(a) {}
  Unary(operation_type const &o, A a) : operation_(o), arg_(a) {}

  operation_type const &operation() const { return operation_;}
  A const &arg() const { return arg_;}

private:
  operation_type operation_;
  A arg_;
};

template <template <typename> class O, typename A>
struct is_expr<Unary<O, A> > { static bool const value = true;};

/// Binary represents a binary operation in an expression.
template <template <typename, typename> class O, typename A1, typename A2>
class Binary
{
  typedef O<typename A1::result_type, typename A2::result_type> operation_type;

public:
  typedef typename operation_type::result_type result_type;

  Binary(A1 a1, A2 a2) : arg1_(a1), arg2_(a2) {}
  Binary(operation_type const &o, A1 a1, A2 a2) : operation_(o), arg1_(a1), arg2_(a2) {}

  operation_type const &operation() const { return operation_;}
  A1 const &arg1() const { return arg1_;}
  A2 const &arg2() const { return arg2_;}

private:
  operation_type operation_;
  A1 arg1_;
  A2 arg2_;
};

namespace impl
{
template <template <typename, typename> class O, typename A1, typename A2,
	  bool E1 = is_expr<A1>::value, bool E2 = is_expr<A2>::value>
struct make_binary;

template <template <typename, typename> class O, typename A1, typename A2>
struct make_binary<O, A1, A2, true, true>
{
  typedef Binary<O, A1, A2> type;
  static type create(A1 a1, A2 a2) { return type(a1, a2);}
};

template <template <typename, typename> class O, typename A1, typename A2>
struct make_binary<O, A1, A2, false, true>
{
  typedef Binary<O, Scalar<A1>, A2> type;
  static type create(A1 a1, A2 a2) { return type(a1, a2);}
};

template <template <typename, typename> class O, typename A1, typename A2>
struct make_binary<O, A1, A2, true, false>
{
  typedef Binary<O, A1, Scalar<A2> > type;
  static type create(A1 a1, A2 a2) { return type(a1, a2);}
};

} // namespace vsip_csl::pi::impl

template <template <typename, typename> class O, typename A1, typename A2>
struct is_expr<Binary<O, A1, A2> > { static bool const value = true;};

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     typename impl::make_binary<expr::op::Add, A1, A2>::type>::type
operator+(A1 a1, A2 a2)
{
  return impl::make_binary<expr::op::Add, A1, A2>::create(a1, a2);
}

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     typename impl::make_binary<expr::op::Sub, A1, A2>::type>::type
operator-(A1 a1, A2 a2)
{
  return impl::make_binary<expr::op::Sub, A1, A2>::create(a1, a2);
}

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     typename impl::make_binary<expr::op::Mult, A1, A2>::type>::type
operator*(A1 a1, A2 a2)
{
  return impl::make_binary<expr::op::Mult, A1, A2>::create(a1, a2);
}

template <typename A1, typename A2>
typename enable_if_c<is_expr<A1>::value || is_expr<A2>::value,
		     typename impl::make_binary<expr::op::Div, A1, A2>::type>::type
operator/(A1 a1, A2 a2)
{
  return impl::make_binary<expr::op::Div, A1, A2>::create(a1, a2);
}

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
