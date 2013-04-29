//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_BLOCK_FILL_HPP
#define VSIP_CORE_BLOCK_FILL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/layout.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/map_traits.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

template <dimension_type Dim,
	  typename       BlockT,
	  typename       OrderT  = typename get_block_layout<BlockT>::order_type,
	  bool           IsGlobal = 
			    Is_global_only<typename BlockT::map_type>::value>
struct Block_fill;

template <dimension_type Dim,
	  typename       BlockT,
	  typename       OrderT>
struct Block_fill<Dim, BlockT, OrderT, true>
{
  typedef typename BlockT::value_type value_type;
  static void exec(BlockT& block, value_type const& val)
  {
    typedef typename Distributed_local_block<BlockT>::type local_block_type;
    typedef typename impl::View_block_storage<local_block_type>::plain_type
		type;

    if (block.map().subblock() != no_subblock)
    {
      // If get_local_block returns a temporary value, we need to copy it.
      // Other (if it returns a reference), this captures it.
      type l_block = get_local_block(block);
      Block_fill<Dim, local_block_type>::exec(l_block, val);
    }
  }
};

template <typename BlockT,
	  typename OrderT>
struct Block_fill<1, BlockT, OrderT, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (index_type i=0; i<block.size(1, 0); ++i)
      block.put(i, val);
  }
};

template <typename BlockT>
struct Block_fill<2, BlockT, row2_type, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type r=0; r<block.size(2, 0); ++r)
      for (vsip::index_type c=0; c<block.size(2, 1); ++c)
	block.put(r, c, val);
  }
};

template <typename BlockT>
struct Block_fill<2, BlockT, col2_type, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type c=0; c<block.size(2, 1); ++c)
      for (vsip::index_type r=0; r<block.size(2, 0); ++r)
	block.put(r, c, val);
  }
};

template <typename BlockT>
struct Block_fill<3, BlockT, tuple<0, 1, 2>, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type z=0; z<block.size(3, 0); ++z)
      for (vsip::index_type y=0; y<block.size(3, 1); ++y)
        for (vsip::index_type x=0; x<block.size(3, 2); ++x)
          block.put(z, y, x, val);
  }
};

template <typename BlockT>
struct Block_fill<3, BlockT, tuple<0, 2, 1>, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type z=0; z<block.size(3, 0); ++z)
      for (vsip::index_type x=0; x<block.size(3, 2); ++x)
        for (vsip::index_type y=0; y<block.size(3, 1); ++y)
          block.put(z, y, x, val);
  }
};

template <typename BlockT>
struct Block_fill<3, BlockT, tuple<1, 2, 0>, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type y=0; y<block.size(3, 1); ++y)
      for (vsip::index_type x=0; x<block.size(3, 2); ++x)
        for (vsip::index_type z=0; z<block.size(3, 0); ++z)
          block.put(z, y, x, val);
  }
};

template <typename BlockT>
struct Block_fill<3, BlockT, tuple<1, 0, 2>, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type y=0; y<block.size(3, 1); ++y)
      for (vsip::index_type z=0; z<block.size(3, 0); ++z)
        for (vsip::index_type x=0; x<block.size(3, 2); ++x)
          block.put(z, y, x, val);
  }
};

template <typename BlockT>
struct Block_fill<3, BlockT, tuple<2, 0, 1>, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type x=0; x<block.size(3, 2); ++x)
      for (vsip::index_type z=0; z<block.size(3, 0); ++z)
        for (vsip::index_type y=0; y<block.size(3, 1); ++y)
          block.put(z, y, x, val);
  }
};

template <typename BlockT>
struct Block_fill<3, BlockT, tuple<2, 1, 0>, false>
{
  typedef typename BlockT::value_type value_type;

  static void exec(BlockT& block, value_type const& val)
  {
    for (vsip::index_type x=0; x<block.size(3, 2); ++x)
      for (vsip::index_type y=0; y<block.size(3, 1); ++y)
        for (vsip::index_type z=0; z<block.size(3, 0); ++z)
          block.put(z, y, x, val);
  }
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_BLOCK_FILL_HPP

