//
// Copyright (c)  2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
