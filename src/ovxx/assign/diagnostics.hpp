//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_assign_diagnostics_hpp_
#define ovxx_assign_diagnostics_hpp_

#include <ovxx/assign_fwd.hpp>
#include <ovxx/expr/scalar.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/type_name.hpp>
#include <sstream>
#include <string>

namespace ovxx
{
namespace assignment
{
namespace detail
{
template <typename T> 
struct assign_type
{
  static std::string name() { return "unknown";}
};

#define OVXX_ASSIGN_TYPE(T)		    \
template <>				    \
struct assign_type<T>			    \
{	                                    \
  static std::string name() { return ""#T;} \
};

OVXX_ASSIGN_TYPE(illegal_mix_of_local_and_global_in_assign)
OVXX_ASSIGN_TYPE(serial_expr)
//OVXX_ASSIGN_TYPE(par_assign)
OVXX_ASSIGN_TYPE(par_expr)
OVXX_ASSIGN_TYPE(par_expr_noreorg)

#undef OVXX_ASSIGN_TYPE

} // namespace ovxx::assignment::detail

template <typename T, dimension_type D>
struct backend_diag
{
  template <typename LHS, typename RHS>
  static std::string info(LHS &, RHS const &)
  {
    std::ostringstream oss;
    oss << "unhandled assignment dispatcher tag:\n"
	<< "  " << detail::assign_type<T>::name();
    return oss.str();
  }
};

template <dimension_type D>
struct backend_diag<serial_expr, D>
{
  template <typename LHS, typename RHS>
  static std::string info(LHS &lhs, RHS const &rhs)
  {
    std::ostringstream oss;
    oss << "  lhs expr: " << type_name<LHS>() << '\n'
	<< "  rhs expr: " << type_name<RHS>() << '\n'
	<< dispatcher::diagnostics<dispatcher::op::assign<D>, void, LHS &, RHS const &>(lhs, rhs);
    return oss.str();
  }
};

template <dimension_type D>
struct backend_diag<par_expr_noreorg, D>
{
  template <typename LHS, typename RHS>
  static std::string info(LHS &lhs, RHS const &rhs)
  {
    std::ostringstream oss;
    if (parallel::has_same_map<D>(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename distributed_local_block<RHS const>::type rhs_local_block_type;
      typedef typename block_traits<lhs_local_block_type>::plain_type lhs_local_ref;
      typedef typename block_traits<rhs_local_block_type const>::plain_type rhs_local_ref;

      lhs_local_ref lhs_local_block = get_local_block(lhs);
      rhs_local_ref rhs_local_block = get_local_block(rhs);

      oss << "LHS and RHS have same map -- local assignment\n"
	  << "  lhs expr: " << type_name(lhs_local_block) << '\n'
	  << "  rhs expr: " << type_name(rhs_local_block) << '\n'
	  << dispatcher::diagnostics<dispatcher::op::assign<D>,
				     void, lhs_local_block_type &, rhs_local_block_type const &>
	(lhs_local_block, rhs_local_block);
    }
    else
    {
      oss << "LHS and RHS have different maps\n"
	  << "error: expr cannot be reorganized";
    }
    return oss.str();
  }
};

template <dimension_type D>
struct backend_diag<par_expr, D>
{
  template <typename LHS, typename RHS>
  static std::string info(LHS &lhs, RHS const &rhs)
  {
    std::ostringstream oss;

    if (parallel::has_same_map<D>(lhs.map(), rhs))
    {
      // Maps are same, no communication required.
      typedef typename distributed_local_block<LHS>::type lhs_local_block_type;
      typedef typename distributed_local_block<RHS>::type rhs_local_block_type;
      typedef typename block_traits<lhs_local_block_type>::plain_type lhs_local_ref;
      typedef typename block_traits<rhs_local_block_type const>::plain_type rhs_local_ref;

      lhs_local_ref lhs_local_block = get_local_block(lhs);
      rhs_local_ref rhs_local_block = get_local_block(rhs);

      oss << "  parallel dim : " << D << "  " << extent<D>(lhs) << '\n'
	  << "  local dim    : " << D << "  " << extent<D>(lhs_local_block) << '\n'
	  << "LHS and RHS have same map -- local assignment\n"
	  << "  lhs expr: " << type_name(lhs_local_block) << '\n'
	  << "  rhs expr: " << type_name(rhs_local_block) << std::endl;
      oss << dispatcher::diagnostics<dispatcher::op::assign<D>,
				     void, lhs_local_block_type &, rhs_local_block_type const &>
	(lhs_local_block, rhs_local_block);
    }
    else
    {
      oss << "LHS and RHS have different maps\n"
	  << "(diagnostics not implemented yet)";
    }
    return oss.str();
  }
};

/// return diagnostics for an assignment
template <template <typename, typename> class V1, typename T1, typename B1,
	  template <typename, typename> class V2, typename T2, typename B2>
std::string diagnostics(V1<T1, B1> lhs_view, V2<T2, B2> rhs_view)
{
  dimension_type const dim = V1<T1, B1>::dim;
  typedef trait<dim, B1, B2 const> d;
  typedef typename d::type assign_type;

  B1 &lhs = lhs_view.block();
  B2 const &rhs = rhs_view.block();

  std::ostringstream oss;

  oss << "--------------------------------------------------------\n"
      << "assignment diagnostics:\n"
      << "  dim : " << dim << ", extent : " << extent<dim>(lhs_view.block()) << '\n';
  // avoid the noise if this is just a simple non-distributed assignment
  if (!is_same<assign_type, serial_expr>::value)
  {
    oss << "  lhs : " << type_name(lhs) << '\n'
	<< "  rhs : " << type_name(rhs) << '\n'
	<< "  is_illegal   : " << (d::is_illegal ? "true" : "false") << '\n'
	<< "  is_rhs_expr  : " << (d::is_rhs_expr ? "true" : "false") << '\n'
	<< "  is_rhs_simple: " << (d::is_rhs_simple ? "true" : "false") << '\n'
	<< "  is_rhs_reorg : " << (d::is_rhs_reorg ? "true" : "false") << '\n'
	<< "  is_lhs_split : " << (d::is_lhs_split ? "true" : "false") << '\n'
	<< "  is_rhs_split : " << (d::is_rhs_split ? "true" : "false") << '\n'
	<< "  lhs_cost     : " << d::lhs_cost << '\n'
	<< "  rhs_cost     : " << d::rhs_cost << '\n'
	<< "  assign type  : " << detail::assign_type<assign_type>::name() << '\n'
	<< "--------------------------------------------------------\n";
  }

  oss << backend_diag<assign_type, dim>::info(lhs, rhs);
  oss << "--------------------------------------------------------" << std::endl;
  return oss.str();
}

/// Alternate form of assign_diagnostics where the right-hand side is a scalar.
template <template <typename, typename> class V, typename T, typename B, typename S>
std::string diagnostics(V<T, B> lhs_view, S scalar)
{
  expr::Scalar<B::dim, S> block(scalar);
  V<S, expr::Scalar<B::dim, S> const> scalar_view(block);
  return diagnostics(lhs_view, scalar_view);
}

} // namespace ovxx::assignment
} // namespace ovxx

#endif
