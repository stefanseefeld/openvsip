/* Copyright (c)  2006 by CodeSourcery.  All rights reserved. */

/// Description
///   Parallel assignment common routines.

#ifndef vsip_core_parallel_assign_common_hpp_
#define vsip_core_parallel_assign_common_hpp_

#include <vsip/support.hpp>
#include <vsip/core/map_fwd.hpp>

namespace vsip
{
namespace impl
{

template <typename MapT>
bool
processor_has_block(MapT const &map,
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

/// Special case for Replicated_map.  Since map is replicated, the answer
/// is always yes (if proc and sb are valid).
template <dimension_type Dim>
bool
processor_has_block(Replicated_map<Dim> const &,
		    processor_type,
		    index_type sb ATTRIBUTE_UNUSED)
{
  assert(sb == 0);
  return true;
}

} // namespace vsip::impl
} // namespace vsip

#endif // vsip_core_parallel_assign_common_hpp_
