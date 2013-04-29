/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/assign_diagnostics.hpp
    @author  Stefan Seefeld
    @date    2009-07-15
    @brief   VSIPL++ Library: Provide block assignment diagnostics.

*/

#ifndef VSIP_OPT_ASSIGN_DIAGNOSTICS_HPP
#define VSIP_OPT_ASSIGN_DIAGNOSTICS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/assign.hpp>
#include <vsip/opt/dispatch_diagnostics.hpp>
#include <vsip/opt/type_name.hpp>
#include <string>

namespace vsip
{
namespace impl
{
namespace assignment
{
template <typename T> 
struct Dispatch_name
{
  static std::string name() { return "unknown";}
};

#define DISPATCH_NAME(T)                    \
template <>				    \
struct Dispatch_name<T>                     \
{	                                    \
  static std::string name() { return ""#T;} \
};

DISPATCH_NAME(illegal_mix_of_local_and_global_in_assign)
DISPATCH_NAME(serial_expr)
//DISPATCH_NAME(par_assign)
DISPATCH_NAME(par_expr)
DISPATCH_NAME(par_expr_noreorg)

#undef DISPATCH_NAME

template <typename T>
struct Dispatcher_traits
{
  template <dimension_type D, typename LHS, typename RHS>
  static void info(LHS &, RHS const &)
  {
    std::cout << "unhandled assignment dispatcher tag:\n"
	      << "  " << Dispatch_name<T>::name() << std::endl;
  }
};

template <>
struct Dispatcher_traits<serial_expr>
{
  template <dimension_type D, typename LHS, typename RHS>
  static void info(LHS &lhs, RHS const &rhs)
  {
    using namespace vsip_csl;
    std::cout << "  lhs expr: " << type_name<LHS>() << '\n'
	      << "  rhs expr: " << type_name<RHS>() << std::endl;
    dispatch_diagnostics<dispatcher::op::assign<D>, void, LHS &, RHS const &>(lhs, rhs);
  }
};

template <>
struct Dispatcher_traits<par_expr_noreorg>
{
  template <dimension_type D, typename LHS, typename RHS>
  static void info(LHS &lhs, RHS const &rhs)
  {
    typedef typename LHS::map_type lhs_map_type;
    if (Is_par_same_map<D, lhs_map_type, RHS>::value(lhs.map(), rhs))
    {
      using namespace vsip_csl;
      // Maps are same, no communication required.
      typedef typename Distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename Distributed_local_block<RHS const>::type rhs_local_block_type;
      typedef typename View_block_storage<lhs_local_block_type>::type::equiv_type
	lhs_storage_type;
      typedef typename View_block_storage<rhs_local_block_type>::type::equiv_type 
	rhs_storage_type;

      lhs_storage_type lhs_local_block = get_local_block(lhs);
      rhs_storage_type rhs_local_block = get_local_block(rhs);

      std::cout << "LHS and RHS have same map -- local assignment\n"
		<< "  lhs expr: " << type_name(lhs_local_block) << '\n'
		<< "  rhs expr: " << type_name(rhs_local_block) << std::endl;
      dispatch_diagnostics<dispatcher::op::assign<D>,
	                   void, lhs_local_block_type &, rhs_local_block_type const &>
	(lhs_local_block, rhs_local_block);
    }
    else
    {
      std::cout << "LHS and RHS have different maps\n"
		<< "error: expr cannot be reorganized" << std::endl;
    }
  }
};

template <>
struct Dispatcher_traits<par_expr>
{
  template <dimension_type D, typename LHS, typename RHS>
  static void info(LHS &lhs, RHS const &rhs)
  {
    using namespace vsip_csl;
    typedef Dispatcher<D, LHS, RHS, par_expr> dispatcher_type;

    typedef typename dispatcher_type::lhs_map_type lhs_map_type;
    typedef typename dispatcher_type::lhs_view_type lhs_view_type;
    typedef typename dispatcher_type::rhs_view_type rhs_view_type;

    if (Is_par_same_map<D, lhs_map_type, RHS>::value(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename Distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename Distributed_local_block<RHS const>::type rhs_local_block_type;
      typedef typename View_block_storage<lhs_local_block_type>::type::equiv_type
	lhs_storage_type;
      typedef typename View_block_storage<rhs_local_block_type>::type::equiv_type
	rhs_storage_type;

      std::cout << "  parallel dim : " << D << "  (" << lhs.size(D, 0);
      for (dimension_type i = 1; i < D; ++i) std::cout << ", " << lhs.size(D, i);
      std::cout << ")\n";

      lhs_storage_type lhs_local_block = get_local_block(lhs);
      rhs_storage_type rhs_local_block = get_local_block(rhs);

      std::cout << "  local dim    : " << D << "  (" << lhs_local_block.size(D, 0);
      for (dimension_type i = 1; i < D; ++i) std::cout << ", " << lhs_local_block.size(D, i);
      std::cout << ")\n"
		<< "LHS and RHS have same map -- local assignment\n"
		<< "  lhs expr: " << type_name(lhs_local_block) << '\n'
		<< "  rhs expr: " << type_name(rhs_local_block) << std::endl;
      dispatch_diagnostics<dispatcher::op::assign<D>,
	                   void, lhs_local_block_type &, rhs_local_block_type const &>
	(lhs_local_block, rhs_local_block);
    }
    else
    {
      std::cout << "LHS and RHS have different maps\n"
		<< "(diagnostics not implemented yet)" << std::endl;
    }
  }
};

} // namespace vsip::impl::assignment

/// print diagnostics for an assignment of the form 'LHS = RHS'.
/// Template parameters:
///   :lhs: left-hand-side view
///   :rhs: right-hand-side view
template <template <typename, typename> class V1, typename T1, typename B1,
	  template <typename, typename> class V2, typename T2, typename B2>
void assign_diagnostics(V1<T1, B1> lhs_view, V2<T2, B2> rhs_view)
{
  dimension_type const dim = V1<T1, B1>::dim;
  typedef assignment::Dispatcher_helper<dim, B1, B2 const> dispatcher;
  typedef typename dispatcher::type dispatch_type;

  B1 &lhs = lhs_view.block();
  B2 const &rhs = rhs_view.block();

  std::cout << "--------------------------------------------------------\n"
	    << "assign diagnostics:\n"
	    << "  dim          : " << dim << "  (" << lhs_view.size(0);
  for (dimension_type i = 1; i < dim; ++i) std::cout << ", " << lhs_view.size(i);
  std::cout << ")\n"
	    << "  LHS    : " << type_name(lhs) << '\n'
	    << "  RHS    : " << type_name(rhs) << '\n'
	    << "  is_illegal   : " << (dispatcher::is_illegal ? "true" : "false") << '\n'
	    << "  is_rhs_expr  : " << (dispatcher::is_rhs_expr ? "true" : "false") << '\n'
	    << "  is_rhs_simple: " << (dispatcher::is_rhs_simple ? "true" : "false") << '\n'
	    << "  is_rhs_reorg : " << (dispatcher::is_rhs_reorg ? "true" : "false") << '\n'
	    << "  is_lhs_split : " << (dispatcher::is_lhs_split ? "true" : "false") << '\n'
	    << "  is_rhs_split : " << (dispatcher::is_rhs_split ? "true" : "false") << '\n'
	    << "  lhs_cost     : " << dispatcher::lhs_cost << '\n'
	    << "  rhs_cost     : " << dispatcher::rhs_cost << '\n'
	    << "  type         : " << assignment::Dispatch_name<dispatch_type>::name() << '\n'
	    << "--------------------------------------------------------\n";

  assignment::Dispatcher_traits<dispatch_type>::template info<dim>(lhs, rhs);
  std::cout << "--------------------------------------------------------" << std::endl;
}

/// Alternate form of assign_diagnostics where the right-hand side is a scalar.
template <template <typename, typename> class V, typename T, typename B, typename S>
void assign_diagnostics(V<T, B> lhs_view, S scalar)
{
  expr::Scalar<B::dim, S> block(scalar);
  V<S, expr::Scalar<B::dim, S> const> scalar_view(block);
  assign_diagnostics(lhs_view, scalar_view);
}


} // namespace vsip::impl
} // namespace vsip

#endif
