/* Copyright (c) 2007 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/map_subset_block.hpp
    @author  Assem Salama
    @date    2007-04-26
    @brief   VSIPL++ Library: Map_subset_block class.
*/

#ifndef VSIP_CORE_EXPR_MAP_SUBSET_BLOCK_HPP
#define VSIP_CORE_EXPR_MAP_SUBSET_BLOCK_HPP


#include <vsip/support.hpp>

namespace vsip
{
namespace impl
{

template <typename Block,
          typename MapT>
class Map_subset_block
{

public:
  static dimension_type const dim = Block::dim;
  typedef typename Block::value_type value_type;
  typedef Local_or_global_map<dim>   map_type;

  // Constructors.
public:
  Map_subset_block(Block& blk, MapT const& map)
    : blk_(blk),
      map_(map),
      sb_(map_.subblock()),
      dom_(map_.template impl_subblock_domain<dim>(sb_))
    {}


  // Accessors.
public:
  length_type size() const VSIP_NOTHROW
  { return dom_.size(); }

  length_type size(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  { assert(block_dim == dim); return dom_[d].size(); }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW { return *(new map_type());}


  value_type get(index_type i) const
  {
    index_type global_i;

    assert(i < dom_[0].size());
    global_i = map_.impl_global_from_local_index(0,sb_,i);
    return blk_.get(global_i);
  }

  value_type get(index_type i, index_type j) const
  {
    index_type global_i, global_j;

    assert(i < dom_[0].size());
    assert(j < dom_[1].size());
    global_i = map_.impl_global_from_local_index(0,sb_,i);
    global_j = map_.impl_global_from_local_index(1,sb_,j);
    return blk_.get(global_i,global_j);
  }

  value_type get(index_type i, index_type j, index_type k) const
  {
    index_type global_i, global_j, global_k;

    assert(i < dom_[0].size());
    assert(j < dom_[1].size());
    assert(k < dom_[2].size());
    global_i = map_.impl_global_from_local_index(0,sb_,i);
    global_j = map_.impl_global_from_local_index(1,sb_,j);
    global_k = map_.impl_global_from_local_index(2,sb_,k);
    return blk_.get(global_i,global_j,global_k);
  }

  // Member data.
private:
  Block&       blk_;
  MapT const&  map_;
  index_type   sb_;
  Domain<dim>  dom_;

};

// Store Distributed_generator_block by reference
template <typename Block, typename MapT>
struct View_block_storage<Map_subset_block<Block, MapT> >
  : By_value_block_storage<Map_subset_block<Block, MapT> >
{};


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_EXPR_MAP_SUBSET_BLOCK_HPP
