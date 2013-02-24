/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression evaluation support

#ifndef vsip_csl_pi_eval_hpp_
#define vsip_csl_pi_eval_hpp_

#include <vsip_csl/pi/expr.hpp>
#include <vsip_csl/pi/reductions.hpp>

namespace vsip_csl
{
namespace pi
{

/// A static visitor to evaluate iterator expressions.
template <typename T> struct Evaluator;

template <int I>
struct Evaluator<Iterator<I> >
{
  typedef index_type result_type;
  static result_type apply(Iterator<I>, index_type i) { return I + i;}
};

template <>
struct Evaluator<Offset>
{
  typedef index_type result_type;
  static result_type apply(Offset const &o, index_type i) { return o + i;}
};

template <>
struct Evaluator<Map>
{
  typedef index_type result_type;
  static result_type apply(Map m, index_type i) { return m.apply(i);}
};

template <typename T> 
struct Evaluator<Scalar<T> >
{
  typedef T result_type;
  static result_type apply(Scalar<T> const &t, index_type i) { return t.value();}
};

template <typename B, typename I> 
struct Evaluator<Call<B, I> >
{
  typedef Call<B, I> call_type;
  typedef typename call_type::result_type result_type;
  static result_type 
  apply(call_type const &c, index_type i)
  { return c.apply(Evaluator<I>::apply(c.i(), i));}
};

template <typename B, typename I> 
struct Evaluator<Call<B, I, whole_domain_type> >
{
  typedef Call<B, I, whole_domain_type> call_type;
  typedef typename call_type::result_type result_type;
  static result_type 
  apply(call_type const &c, index_type i)
  { return c.apply(Evaluator<I>::apply(c.i(), i));}
};

template <typename B, typename J> 
struct Evaluator<Call<B, whole_domain_type, J> >
{
  typedef Call<B, whole_domain_type, J> call_type;
  typedef typename call_type::result_type result_type;
  static result_type 
  apply(call_type const &c, index_type j) 
  { return c.apply(Evaluator<J>::apply(c.j(), j));}
};

template <template <typename> class O, typename A> 
struct Evaluator<Unary<O, A> >
{
  typedef O<typename Evaluator<A>::result_type> operation_type;
  typedef typename operation_type::result_type result_type;
  static result_type
  apply(Unary<O, A> const &e, index_type i)
  {
    operation_type const &op = e.operation();
    return op(Evaluator<A>::apply(e.arg(), i));
  }
};

template <template <typename, typename> class O, typename A1, typename A2> 
struct Evaluator<Binary<O, A1, A2> >
{
  typedef O<typename Evaluator<A1>::result_type,
	    typename Evaluator<A2>::result_type> operation_type;
  typedef typename operation_type::result_type result_type;
  static result_type
  apply(Binary<O, A1, A2> const &e, index_type i)
  {
    operation_type const &op = e.operation();
    return op(Evaluator<A1>::apply(e.arg1(), i),
	      Evaluator<A2>::apply(e.arg2(), i));
  }
};

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
