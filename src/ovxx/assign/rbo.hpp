//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_assign_rbo_hpp_
#define ovxx_assign_rbo_hpp_

#include <ovxx/expr/evaluate.hpp>

namespace ovxx
{
namespace dispatcher
{

template <dimension_type D, typename LHS,
	  template <typename> class O,
	  typename B>
struct Evaluator<op::assign<D>, be::rbo_expr,
		 void(LHS &, expr::Unary<O, B> const &)>
{
  typedef expr::Unary<O, B> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static bool const ct_valid = true;
  static char const *name() { return OVXX_DISPATCH_EVAL_NAME;}
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs) { rhs.apply(lhs);}
};

template <dimension_type D, typename LHS,
	  template <typename, typename> class O,
	  typename B1, typename B2>
struct Evaluator<op::assign<D>, be::rbo_expr,
		 void(LHS &, expr::Binary<O, B1, B2> const &)>
{
  typedef expr::Binary<O, B1, B2> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static bool const ct_valid = true;
  static char const *name() { return OVXX_DISPATCH_EVAL_NAME;}
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs) { rhs.apply(lhs);}
};

template <dimension_type D, typename LHS,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
struct Evaluator<op::assign<D>, be::rbo_expr,
		 void(LHS &, expr::Ternary<O, B1, B2, B3> const &)>
{
  typedef expr::Ternary<O, B1, B2, B3> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static bool const ct_valid = true;
  static char const *name() { return OVXX_DISPATCH_EVAL_NAME;}
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs) { rhs.apply(lhs);}
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
