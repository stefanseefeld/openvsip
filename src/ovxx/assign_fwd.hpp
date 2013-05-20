//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_assign_fwd_hpp_
#define ovxx_assign_fwd_hpp_

#include <ovxx/block_traits.hpp>
#include <ovxx/parallel/map_traits.hpp>
#include <ovxx/dispatch.hpp>

namespace ovxx
{
namespace assignment
{

struct serial_expr;
template <typename P> struct par_assign;
struct par_expr_noreorg;
struct par_expr;
struct illegal_mix_of_local_and_global_in_assign;

/// Determine the proper tag for handling an assignment:
///
/// * if LHS and RHS are serial, use serial
/// * else if LHS and RHS are simple distributed, use par_assign
/// * else (LHS and RHS are distributed expression) use par_expr
template <dimension_type D, typename LHS, typename RHS,
	  bool EarlyBinding = false>
struct trait;

template <dimension_type D, typename LHS, typename RHS,
	  typename T = typename trait<D, LHS, RHS, false>::type>
struct Dispatcher;

} // namespace ovxx::assignment

namespace dispatcher
{
namespace op
{
template <dimension_type D> struct assign;
}

template <dimension_type D>
struct List<op::assign<D> >
{
  typedef make_type_list<be::user,
			 be::cuda,
			 be::dense_expr,
			 be::copy,
			 be::op_expr,
			 be::simd,
			 be::fc_expr,
			 be::rbo_expr,
			 be::mdim_expr,
			 be::loop_fusion>::type type;
};


} // namespace vsip_csl::dispatcher

template <dimension_type D, typename LHS, typename RHS>
inline void 
assign(LHS &, RHS const &);

} // namespace ovxx

#endif
