//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_map_subset_block_hpp_
#define ovxx_parallel_map_subset_block_hpp_

#include <vsip/support.hpp>

namespace ovxx
{
namespace parallel
{

template <typename B, typename M>
class map_subset_block
{

public:
  static dimension_type const dim = B::dim;
  typedef typename B::value_type value_type;
  typedef local_or_global_map<dim> map_type;

  map_subset_block(B &block, M const &map)
    : block_(block),
      map_(map),
      sb_(map_.subblock()),
      dom_(map_.template impl_subblock_domain<dim>(sb_))
    {}

  length_type size() const VSIP_NOTHROW { return dom_.size();}
  length_type size(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(block_dim == dim); return dom_[d].size();}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const &map() const VSIP_NOTHROW { return *(new map_type());}

  value_type get(index_type i) const
  {
    index_type global_i;

    OVXX_PRECONDITION(i < dom_[0].size());
    global_i = map_.impl_global_from_local_index(0,sb_,i);
    return block_.get(global_i);
  }

  value_type get(index_type i, index_type j) const
  {
    index_type global_i, global_j;

    OVXX_PRECONDITION(i < dom_[0].size());
    OVXX_PRECONDITION(j < dom_[1].size());
    global_i = map_.impl_global_from_local_index(0,sb_,i);
    global_j = map_.impl_global_from_local_index(1,sb_,j);
    return block_.get(global_i,global_j);
  }

  value_type get(index_type i, index_type j, index_type k) const
  {
    OVXX_PRECONDITION(i < dom_[0].size());
    OVXX_PRECONDITION(j < dom_[1].size());
    OVXX_PRECONDITION(k < dom_[2].size());
    index_type global_i = map_.impl_global_from_local_index(0,sb_,i);
    index_type global_j = map_.impl_global_from_local_index(1,sb_,j);
    index_type global_k = map_.impl_global_from_local_index(2,sb_,k);
    return block_.get(global_i,global_j,global_k);
  }

private:
  B &block_;
  M const &map_;
  index_type   sb_;
  Domain<dim>  dom_;

};

} // namespace ovxx::parallel

template <typename B, typename M>
struct block_traits<parallel::map_subset_block<B, M> >
  : by_value_traits<parallel::map_subset_block<B, M> >
{};

} // namespace ovxx

#endif
