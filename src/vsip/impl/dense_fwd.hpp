//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_dense_fwd_hpp_
#define vsip_impl_dense_fwd_hpp_

#include <vsip/support.hpp>

namespace vsip
{

/// Dense block, as defined in standard [view.dense].
///
/// A Dense block is a modifiable, allocatable 1-dimensional block
/// or 1,x-dimensional block, for a fixed x, that explicitly stores
/// one value for each Index in its domain.
template <dimension_type D = 1,
	  typename       T = VSIP_DEFAULT_VALUE_TYPE,
	  typename       O = typename ovxx::row_major<D>::type,
	  typename       M = Local_map>
class Dense;

} // namespace vsip

#endif
