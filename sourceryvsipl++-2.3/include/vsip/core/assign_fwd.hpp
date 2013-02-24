/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/assign_fwd.hpp
    @author  Stefan Seefeld
    @date    2009-07-15
    @brief   VSIPL++ Library: view / block assignment forward declarations.

    Separating declarations here allows eval/diag.hpp to be used
    from headers that are included by assign.hpp, for
    example, in eval_fastconv.hpp.

*/

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
assign(LHS &, RHS &);

} // namespace vsip::impl
} // namespace vsip

#endif
