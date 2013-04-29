//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PARALLEL_TRANSPOSE_MAP_HPP
#define VSIP_CORE_PARALLEL_TRANSPOSE_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/parallel/transpose_map_decl.hpp>
#include <vsip/core/vector.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{

template <typename MapT>
const_Vector<processor_type>
Transpose_map<MapT>::processor_set()
  const
{
  return map_.processor_set();
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_PARALLEL_TRANSPOSE_MAP_HPP
