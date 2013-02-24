/* Copyright (c)  2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/assign_common.hpp
    @author  Jules Bergmann
    @date    2006-08-29
    @brief   VSIPL++ Library: Parallel assignment common routines.

*/

#ifndef VSIP_CORE_PARALLEL_ASSIGN_COMMON_HPP
#define VSIP_CORE_PARALLEL_ASSIGN_COMMON_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/map_fwd.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

template <typename MapT>
bool
processor_has_block(
  MapT const&    map,
  processor_type proc,
  index_type     sb)
{
  typedef typename MapT::processor_iterator iterator;

  for (iterator cur = map.processor_begin(sb);
       cur != map.processor_end(sb);
       ++cur)
  {
    if (*cur == proc)
      return true;
  }
  return false;
}



// Special case for Global_map.  Since map is replicated, the answer
// is always yes (if proc and sb are valid).

template <dimension_type Dim>
bool
processor_has_block(
  Global_map<Dim> const& /*map*/,
  processor_type         /*proc*/,
  index_type             sb)
{
  assert(sb == 0);
  return true;
}


} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_IMPL_PAR_ASSIGN_COMMON_HPP
