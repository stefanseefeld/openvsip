//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_assign_hpp_
#define ovxx_parallel_assign_hpp_

#include <ovxx/support.hpp>
#include <ovxx/config.hpp>
#include <ovxx/parallel/assign_fwd.hpp>
#include <ovxx/parallel/assign_chain.hpp>
#include <ovxx/parallel/assign_block_vector.hpp>
#include <ovxx/parallel/expr.hpp>

namespace ovxx
{
namespace assignment
{
/// Specialization for parallel assignment, RHS is simple (A = B)
template <dimension_type D, typename LHS, typename RHS, typename Assign>
struct Dispatcher<D, LHS, RHS, par_assign<Assign> >
{
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;  
  typedef typename view_of<LHS>::type lhs_view_type;
  typedef typename view_of<RHS>::const_type rhs_view_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    if (parallel::has_same_map<D>(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename block_traits<lhs_local_block_type>::plain_type 
	lhs_local_storage_type;
      typedef typename block_traits<rhs_local_block_type>::plain_type
	rhs_local_storage_type;

      lhs_local_storage_type lhs_local_block = get_local_block(lhs);
      rhs_local_storage_type rhs_local_block = get_local_block(rhs);

      Dispatcher<D, lhs_local_block_type, rhs_local_block_type>::exec(lhs_local_block,
								      rhs_local_block);
    }
    else
    {
      lhs_view_type lhs_view(lhs);
      rhs_view_type rhs_view(const_cast<RHS&>(rhs));
      parallel::Assignment<D, LHS, RHS, Assign> pa(lhs_view, rhs_view);
      pa();
    }
  }
};

/// Specialization for distributed expressions where the RHS can be
/// reorganized.
template <dimension_type D, typename LHS, typename RHS>
struct Dispatcher<D, LHS, RHS, par_expr>
{
  typedef typename view_of<LHS>::type lhs_view_type;
  typedef typename view_of<RHS>::const_type rhs_view_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    if (parallel::has_same_map<D>(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename block_traits<lhs_local_block_type>::plain_type 
	lhs_local_storage_type;
      typedef typename block_traits<rhs_local_block_type>::plain_type 
	rhs_local_storage_type;

      lhs_local_storage_type lhs_local_block = get_local_block(lhs);
      rhs_local_storage_type rhs_local_block = get_local_block(rhs);

      Dispatcher<D, lhs_local_block_type, rhs_local_block_type>::exec(lhs_local_block,
								      rhs_local_block);
    }
    else
    {
      // Maps are different, fall out to general expression.
      lhs_view_type lhs_view(lhs);
      rhs_view_type rhs_view(const_cast<RHS&>(rhs));
      parallel::expr(lhs_view, rhs_view);
    }
  }
};

/// Specialization for distributed expressions that cannot be reorganized.
/// 
/// Types of expressions that cannot be reorganized:
///  - Vmmul_expr_blocks
template <dimension_type D, typename LHS, typename RHS>
struct Dispatcher<D, LHS, RHS, par_expr_noreorg>
{
  typedef typename view_of<LHS>::type lhs_view_type;
  typedef typename view_of<RHS>::const_type rhs_view_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    if (parallel::has_same_map<D>(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename block_traits<lhs_local_block_type>::plain_type 
	lhs_local_storage_type;
      typedef typename block_traits<rhs_local_block_type>::plain_type
	rhs_local_storage_type;

      lhs_local_storage_type lhs_local_block = get_local_block(lhs);
      rhs_local_storage_type rhs_local_block = get_local_block(rhs);

      Dispatcher<D, lhs_local_block_type, rhs_local_block_type>::exec(lhs_local_block,
								      rhs_local_block);
    }
    else
    {
      OVXX_DO_THROW(unimplemented("Expression cannot be reorganized"));
    }
  }
};
} // namespace ovxx::assignment
} // namespace ovxx

#endif
