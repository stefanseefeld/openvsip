/* Copyright (c)  2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/par_assign_fwd.hpp
    @author  Jules Bergmann
    @date    2006-07-14
    @brief   VSIPL++ Library: Parallel assignment class.

*/

#ifndef VSIP_IMPL_PAR_ASSIGN_FWD_HPP
#define VSIP_IMPL_PAR_ASSIGN_FWD_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

struct Chained_assign;
struct Blkvec_assign;
struct Pas_assign;
struct Pas_assign_eb;
struct Direct_pas_assign;

// Parallel assignment.
template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       Block1,
	  typename       Block2,
	  typename       ImplTag>
class Par_assign;



} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_IMPL_PAR_ASSIGN_FWD_HPP
