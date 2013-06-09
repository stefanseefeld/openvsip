//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_assign_hpp_
#define ovxx_assign_hpp_

#include <ovxx/ct_assert.hpp>
#include <ovxx/assign_fwd.hpp>
#include <ovxx/assign/diagnostics.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/parallel/assign_fwd.hpp>
#include <ovxx/dda.hpp>
#include <ovxx/assign/loop_fusion.hpp>
#ifdef OVXX_PARALLEL
# include <ovxx/parallel/map_traits.hpp>
# include <ovxx/parallel/expr.hpp>
# include <ovxx/parallel/assign.hpp>
#endif

namespace ovxx
{
namespace assignment
{

/// Apply a small meta-program to determine the proper tag for handling
/// an assignment:
///
///    if LHS and RHS are serial, use serial
///    else if LHS and RHS are simple distributed, use par_assign
///    else (LHS and RHS are distributed expression) use par_expr
template <dimension_type D, typename LHS, typename RHS, bool EarlyBinding>
struct trait
{
  typedef typename LHS::map_type lhs_map_type;
  typedef typename RHS::map_type rhs_map_type;

  // Cannot mix local and distributed data in expressions.
  static bool const is_illegal    =
    !((parallel::is_local_map<lhs_map_type>::value && 
       parallel::is_local_map<rhs_map_type>::value) ||
      (parallel::is_global_map<lhs_map_type>::value &&
       parallel::is_global_map<rhs_map_type>::value));

  static bool const is_local      = parallel::is_local_map<lhs_map_type>::value &&
    parallel::is_local_map<rhs_map_type>::value;
  static bool const is_rhs_expr   = is_expr_block<RHS>::value;
  static bool const is_rhs_simple = is_simple_distributed_block<RHS>::value;
  static bool const is_rhs_reorg  = parallel::is_reorg_ok<RHS>::value;

  static bool const is_lhs_split  = is_split_block<LHS>::value;
  static bool const is_rhs_split  = is_split_block<RHS>::value;

  static int const  lhs_cost      = dda::Data<LHS, dda::out>::ct_cost;
  static int const  rhs_cost      = dda::Data<RHS, dda::in>::ct_cost;

  typedef typename
  parallel::choose_par_assign_impl<D, LHS, RHS, EarlyBinding>::type par_assign_type;

  typedef 
  typename conditional<is_illegal, illegal_mix_of_local_and_global_in_assign,
  typename conditional<is_local, serial_expr,
  typename conditional<is_rhs_simple, par_assign<par_assign_type>,
  typename conditional<is_rhs_reorg, par_expr,
                       /* else */ par_expr_noreorg>::type
	              >::type 
	              >::type
	              >::type type;
};


/// Specialization for serial, 1-dimensional assignment.
template <typename LHS, typename RHS>
struct Dispatcher<1, LHS, RHS, serial_expr>
{
  typedef typename get_block_layout<LHS>::type lhs_layout_type;
  typedef typename get_block_layout<RHS>::type rhs_layout_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    dispatch<dispatcher::op::assign<1>, void, LHS &, RHS const &>(lhs, rhs);
  }
};

/// Specialization for serial, 2-dimensional assignment.
template <typename LHS, typename RHS>
struct Dispatcher<2, LHS, RHS, serial_expr>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    dispatch<dispatcher::op::assign<2>, void, LHS &, RHS const &>(lhs, rhs);
  }
};

/// Specialization for serial, 3-dimensional assignment.
template <typename LHS, typename RHS>
struct Dispatcher<3, LHS, RHS, serial_expr>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    dispatch<dispatcher::op::assign<3>, void, LHS &, RHS const &>(lhs, rhs);
  }
};

} // namespace ovxx::assignment

/// Assign one block to another.
/// This process involves a fairly sophisticated multi-dispatch logic.
template <dimension_type D, typename LHS, typename RHS>
inline void 
assign(LHS &lhs, RHS const &rhs)
{
  typedef typename
    conditional<is_expr_block<RHS>::value,
		RHS const,
		typename remove_const<RHS>::type>::type
    rhs_type;
  assignment::Dispatcher<D, LHS, rhs_type>::exec(lhs, rhs);
}

} // namespace ovxx

#endif
