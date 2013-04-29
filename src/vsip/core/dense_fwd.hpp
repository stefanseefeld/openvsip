//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_DENSE_FWD_HPP
#define VSIP_CORE_DENSE_FWD_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// Dense block, as defined in standard [view.dense].
///
/// A Dense block is a modifiable, allocatable 1-dimensional block
/// or 1,x-dimensional block, for a fixed x, that explicitly stores
/// one value for each Index in its domain.
template <dimension_type Dim      = 1,
	  typename       T        = VSIP_DEFAULT_VALUE_TYPE,
	  typename       Order    = typename impl::Row_major<Dim>::type,
	  typename       Map      = Local_map>
class Dense;

} // namespace vsip

#endif // VSIP_IMPL_DENSE_FWD_HPP
