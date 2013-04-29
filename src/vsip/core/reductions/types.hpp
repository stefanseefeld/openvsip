/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/reductions/types.hpp
    @author  Jules Bergmann
    @date    2006-01-10
    @brief   VSIPL++ Library: Enumeration type for reduction functions.
	     [math.fns.reductions].

*/

#ifndef VSIP_IMPL_REDUCTIONS_TYPES_HPP
#define VSIP_IMPL_REDUCTIONS_TYPES_HPP

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

enum reduction_type
{
  reduce_all_true,
  reduce_all_true_bool,
  reduce_any_true,
  reduce_any_true_bool,
  reduce_mean,
  reduce_mean_magsq,
  reduce_sum,
  reduce_sum_bool,
  reduce_sum_sq,
  reduce_max_magsq,
  reduce_max_mag,
  reduce_min_magsq,
  reduce_min_mag,
  reduce_max,
  reduce_min
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_REDUCTIONS_TYPES_HPP
