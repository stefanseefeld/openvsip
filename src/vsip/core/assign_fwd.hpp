//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_ASSIGN_FWD_HPP
#define VSIP_CORE_ASSIGN_FWD_HPP

namespace vsip
{
namespace impl
{
namespace assignment
{

struct illegal_mix_of_local_and_global_in_assign;
struct serial_expr;
template <typename ParAssignImpl> struct par_assign;
struct par_expr_noreorg;
struct par_expr;

template <dimension_type Dim,
	  typename       Block1,
	  typename       Block2,
	  bool           EarlyBinding = false>
struct Dispatcher_helper;


template <dimension_type Dim,
	  typename       Block1,
	  typename       Block2,
	  typename       Tag
	  = typename Dispatcher_helper<Dim, Block1, Block2, false>::type>
struct Dispatcher;

} // namespace vsip::impl::assignment

template <dimension_type D, typename LHS, typename RHS>
inline void 
assign(LHS &, RHS const &);

} // namespace vsip::impl
} // namespace vsip

#endif
