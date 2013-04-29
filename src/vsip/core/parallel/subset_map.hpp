/* Copyright (c) 2006, 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/subset_map.hpp
    @author  Jules Bergmann
    @date    2006-12-10
    @brief   VSIPL++ Library: Map class for distributed subsets.

*/

#ifndef VSIP_CORE_PARALLEL_SUBSET_MAP_HPP
#define VSIP_CORE_PARALLEL_SUBSET_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/parallel/subset_map_decl.hpp>
#include <vsip/core/vector.hpp>



/***********************************************************************
  Declarations & Class Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{

template <dimension_type Dim>
const_Vector<processor_type>
Subset_map<Dim>::processor_set() const
{
  Vector<processor_type> pset(this->num_processors());

  for (index_type i=0; i<this->num_processors(); ++i)
    pset.put(i, this->pvec_[i]);

  return pset;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_PARALLEL_SUBSET_MAP_HPP
