/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/assign.hpp
    @author  Stefan Seefeld
    @date    2009-07-15
    @brief   VSIPL++ Library: view / block assignment.

*/

#ifndef VSIP_CORE_ASSIGN_HPP
#define VSIP_CORE_ASSIGN_HPP

#include <vsip/core/static_assert.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/map_traits.hpp>
#include <vsip/core/parallel/expr.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/expr/assign.hpp>
#endif
#include <vsip/core/parallel/assign.hpp>
#include <vsip/core/assign_fwd.hpp>
#include <vsip/core/c++0x.hpp>
#include <cstring> // for memcpy

namespace vsip
{
namespace impl
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
struct Dispatcher_helper
{
  typedef typename LHS::map_type lhs_map_type;
  typedef typename RHS::map_type rhs_map_type;

  // Cannot mix local and distributed data in expressions.
  static bool const is_illegal    =
    !((Is_local_map<lhs_map_type>::value && Is_local_map<rhs_map_type>::value) ||
      (Is_global_map<lhs_map_type>::value && Is_global_map<rhs_map_type>::value));

  static bool const is_local      = Is_local_map<lhs_map_type>::value &&
                                    Is_local_map<rhs_map_type>::value;
  static bool const is_rhs_expr   = Is_expr_block<RHS>::value;
  static bool const is_rhs_simple = Is_simple_distributed_block<RHS>::value;
  static bool const is_rhs_reorg  = Is_par_reorg_ok<RHS>::value;

  static bool const is_lhs_split  = Is_split_block<LHS>::value;
  static bool const is_rhs_split  = Is_split_block<RHS>::value;

  static int const  lhs_cost      = Ext_data_cost<LHS>::value;
  static int const  rhs_cost      = Ext_data_cost<RHS>::value;

  typedef typename
  Choose_par_assign_impl<D, LHS, RHS, EarlyBinding>::type par_assign_type;

  typedef 
  typename conditional<is_illegal,    illegal_mix_of_local_and_global_in_assign,
  typename conditional<is_local,      serial_expr,
  typename conditional<is_rhs_simple, par_assign<par_assign_type>,
  typename conditional<is_rhs_reorg,  par_expr,
                       /* else */     par_expr_noreorg>::type
	              >::type 
	              >::type
	              >::type type;
};

/// Specialization for serial, 1-dimensional assignment.
template <typename LHS, typename RHS>
struct Dispatcher<1, LHS, RHS, serial_expr>
{
  typedef typename Block_layout<LHS>::layout_type lhs_layout_type;
  typedef typename Block_layout<RHS>::layout_type rhs_layout_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
#ifdef VSIP_IMPL_REF_IMPL
    length_type const size = lhs.size(1, 0);
    for (index_type i=0; i<size; ++i) lhs.put(i, rhs.get(i));
#else
    using namespace vsip_csl;
    dispatch<dispatcher::op::assign<1>, void, LHS &, RHS const &>(lhs, rhs);
#endif
  }
};

/// Specialization for serial, 2-dimensional assignment.
template <typename LHS, typename RHS>
struct Dispatcher<2, LHS, RHS, serial_expr>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
#ifdef VSIP_IMPL_REF_IMPL
    length_type const rows = lhs.size(2, 0);
    length_type const cols = lhs.size(2, 1);
    for (index_type i=0; i<rows; ++i)
      for (index_type j=0; j<cols; ++j)
	lhs.put(i, j, rhs.get(i, j));
#else
    using namespace vsip_csl;
    dispatch<dispatcher::op::assign<2>, void, LHS &, RHS const &>(lhs, rhs);
#endif
  }
};

/// Specialization for serial, 3-dimensional assignment.
template <typename LHS, typename RHS>
struct Dispatcher<3, LHS, RHS, serial_expr>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
#ifdef VSIP_IMPL_REF_IMPL
    length_type const size0 = lhs.size(3, 0);
    length_type const size1 = lhs.size(3, 1);
    length_type const size2 = lhs.size(3, 2);

    for (index_type i=0; i<size0; ++i)
      for (index_type j=0; j<size1; ++j)
        for (index_type k=0; k<size2; ++k)
          lhs.put(i, j, k, rhs.get(i, j, k));
#else
    using namespace vsip_csl;
    dispatch<dispatcher::op::assign<3>, void, LHS &, RHS const &>(lhs, rhs);
#endif
  }
};

/// Specialization for parallel assignment, RHS is simple (A = B)
template <dimension_type D, typename LHS, typename RHS, typename Assign>
struct Dispatcher<D, LHS, RHS, par_assign<Assign> >
{
  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;  
  typedef typename LHS::map_type lhs_map_type;
  typedef typename View_of_dim<D, lhs_value_type, LHS>::type lhs_view_type;
  typedef typename View_of_dim<D, rhs_value_type, RHS>::const_type rhs_view_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    if (Is_par_same_map<D, lhs_map_type, RHS>::value(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename Distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename Distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename 
	View_block_storage<lhs_local_block_type>::type::equiv_type 
	lhs_local_storage_type;
      typedef typename 
	View_block_storage<rhs_local_block_type>::type::equiv_type
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
      Par_assign<D, lhs_value_type, rhs_value_type, LHS, RHS, Assign>
	pa(lhs_view, rhs_view);
      pa();
    }
  }
};

/// Specialization for distributed expressions where the RHS can be
/// reorganized.
template <dimension_type D, typename LHS, typename RHS>
struct Dispatcher<D, LHS, RHS, par_expr>
{
  typedef typename LHS::map_type lhs_map_type;
  typedef typename 
  View_of_dim<D, typename LHS::value_type, LHS>::type lhs_view_type;
  typedef typename 
  View_of_dim<D, typename RHS::value_type, RHS>::const_type rhs_view_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    if (Is_par_same_map<D, lhs_map_type, RHS>::value(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename Distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename Distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename 
	View_block_storage<lhs_local_block_type>::type::equiv_type 
	lhs_local_storage_type;
      typedef typename 
	View_block_storage<rhs_local_block_type>::type::equiv_type 
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
      impl::par_expr(lhs_view, rhs_view);
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
  typedef typename LHS::map_type lhs_map_type;

  typedef typename 
  View_of_dim<D, typename LHS::value_type, LHS>::type lhs_view_type;
  typedef typename 
  View_of_dim<D, typename RHS::value_type, RHS>::const_type rhs_view_type;

  static void exec(LHS &lhs, RHS const &rhs)
  {
    if (Is_par_same_map<D, lhs_map_type, RHS>::value(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename Distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename Distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename 
	View_block_storage<lhs_local_block_type>::type::equiv_type 
	lhs_local_storage_type;
      typedef typename 
	View_block_storage<rhs_local_block_type>::type::equiv_type
	rhs_local_storage_type;

      lhs_local_storage_type lhs_local_block = get_local_block(lhs);
      rhs_local_storage_type rhs_local_block = get_local_block(rhs);

      Dispatcher<D, lhs_local_block_type, rhs_local_block_type>::exec(lhs_local_block,
								      rhs_local_block);
    }
    else
    {
      VSIP_IMPL_THROW(impl::unimplemented("Expression cannot be reorganized"));
    }
  }
};
} // namespace vsip::impl::assignment

/// Assign one block to another.
/// This process involves a fairly involved multi-dispatch logic.
template <dimension_type D, typename LHS, typename RHS>
inline void 
assign(LHS &lhs, RHS &rhs)
{
  assignment::Dispatcher<D, LHS, RHS>::exec(lhs, rhs);
}

} // namespace vsip::impl
} // namespace vsip

#endif
