//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_subset_map_hpp_
#define ovxx_parallel_subset_map_hpp_

#include <ovxx/parallel/subset_map_decl.hpp>
#include <vsip/impl/vector.hpp>

namespace ovxx
{
namespace parallel
{

template <dimension_type D>
const_Vector<processor_type>
subset_map<D>::processor_set() const
{
  Vector<processor_type> pset(this->num_processors());

  for (index_type i=0; i<this->num_processors(); ++i)
    pset.put(i, this->pvec_[i]);

  return pset;
}

} // namespace ovxx::parallel
} // namespace ovxx

#endif
