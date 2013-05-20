//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_assign_loop_fusion_hpp_
#define ovxx_assign_loop_fusion_hpp_

#include <ovxx/expr/evaluate.hpp>

namespace ovxx
{
namespace assignment
{

template <typename LHS, typename RHS,
	  dimension_type D = LHS::dim,
	  typename O = typename get_block_layout<LHS>::order_type>
struct loop_fusion;

template <typename LHS, typename RHS, typename O>
struct loop_fusion<LHS, RHS, 1, O>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);
    length_type const size = lhs.size(1, 0);
    for (index_type i=0; i<size; ++i)
      lhs.put(i, rhs.get(i));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 2, row2_type>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const rows = lhs.size(2, 0);
    length_type const cols = lhs.size(2, 1);
    for (index_type i=0; i<rows; ++i)
      for (index_type j=0; j<cols; ++j)
	lhs.put(i, j, rhs.get(i, j));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 2, col2_type>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const rows = lhs.size(2, 0);
    length_type const cols = lhs.size(2, 1);
    for (index_type j=0; j<cols; ++j)
      for (index_type i=0; i<rows; ++i)
	lhs.put(i, j, rhs.get(i, j));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 3, tuple<0,1,2> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type i=0; i<size0; ++i)
      for (index_type j=0; j<size1; ++j)
	for (index_type k=0; k<size2; ++k)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 3, tuple<0,2,1> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type i=0; i<size0; ++i)
      for (index_type k=0; k<size2; ++k)
	for (index_type j=0; j<size1; ++j)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 3, tuple<1,0,2> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type j=0; j<size1; ++j)
      for (index_type i=0; i<size0; ++i)
	for (index_type k=0; k<size2; ++k)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 3, tuple<1,2,0> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type j=0; j<size1; ++j)
      for (index_type k=0; k<size2; ++k)
	for (index_type i=0; i<size0; ++i)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 3, tuple<2,0,1> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type k=0; k<size2; ++k)
      for (index_type i=0; i<size0; ++i)
	for (index_type j=0; j<size1; ++j)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

template <typename LHS, typename RHS>
struct loop_fusion<LHS, RHS, 3, tuple<2,1,0> >
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    expr::evaluate(rhs);

    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type k=0; k<size2; ++k)
      for (index_type j=0; j<size1; ++j)
	for (index_type i=0; i<size0; ++i)
	  lhs.put(i, j, k, rhs.get(i, j, k));
  }
};

} // namespace ovxx::assignment

namespace dispatcher
{
template <dimension_type D, typename LHS, typename RHS>
struct Evaluator<op::assign<D>, be::loop_fusion, void(LHS &, RHS const &)>
{
  static bool const ct_valid = true;
  static std::string name() { return OVXX_DISPATCH_EVAL_NAME;}
  static bool rt_valid(LHS &, RHS const &) { return true;}  
  static void exec(LHS &lhs, RHS const &rhs)
  { assignment::loop_fusion<LHS, RHS, D>::exec(lhs, rhs);}
};

} // namespace ovxx::dispatcher
} // namespace ovxx


#endif
