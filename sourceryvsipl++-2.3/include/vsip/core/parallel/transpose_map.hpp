/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/transpose_map.hpp
    @author  Jules Bergmann
    @date    2008-05-30
    @brief   VSIPL++ Library: Map class for transposes.

*/

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
