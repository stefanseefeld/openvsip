/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/operations.hpp
    @author  Stefan Seefeld
    @date    2005-02-15
    @brief   VSIPL++ Library: Elementwise operations.
*/

#ifndef VSIP_CORE_EXPR_OPERATIONS_HPP
#define VSIP_CORE_EXPR_OPERATIONS_HPP

#include <vsip/core/promote.hpp>

namespace vsip_csl
{
namespace expr
{
namespace op
{

template <typename Operand>
struct Plus
{
  typedef Operand result_type;

  static char const* name() { return "+"; }
  static result_type apply(Operand op) { return op;}
  result_type operator()(Operand op) const { return apply(op);}
};

template <typename Operand>
struct Minus
{
  typedef Operand result_type;

  static char const* name() { return "-"; }
  static result_type apply(Operand op) { return -op;}
  result_type operator()(Operand op) const { return apply(op);}
};

template <typename LType, typename RType>
struct Add
{
  typedef typename vsip::Promotion<LType, RType>::type result_type;

  static char const* name() { return "+"; }
  static result_type apply(LType lhs, RType rhs) { return lhs + rhs;}
  result_type operator()(LType lhs, RType rhs) const { return apply(lhs, rhs);}
};

template <typename LType, typename RType>
struct Sub
{
  typedef typename vsip::Promotion<LType, RType>::type result_type;

  static char const* name() { return "-"; }
  static result_type apply(LType lhs, RType rhs) { return lhs - rhs;}
  result_type operator()(LType lhs, RType rhs) const { return apply(lhs, rhs);}
};

template <typename LType, typename RType>
struct Mult
{
  typedef typename vsip::Promotion<LType, RType>::type result_type;

  static char const* name() { return "*"; }
  static result_type apply(LType lhs, RType rhs) { return lhs * rhs;}
  result_type operator()(LType lhs, RType rhs) const { return apply(lhs, rhs);}
};

template <typename LType, typename RType>
struct Div
{
  typedef typename vsip::Promotion<LType, RType>::type result_type;

  static char const* name() { return "/"; }
  static result_type apply(LType lhs, RType rhs) { return lhs / rhs;}
  result_type operator()(LType lhs, RType rhs) const { return apply(lhs, rhs);}
};

} // namespace vsip_csl::expr::op
} // namespace vsip_csl::expr
} // namespace vsip_csl

#endif
