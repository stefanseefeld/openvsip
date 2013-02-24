/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/us-block.cpp
    @author  Jules Bergmann
    @date    2006-01-31
    @brief   VSIPL++ Library: Unit tests for Us_block's.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>
#include <vsip/support.hpp>
#include <vsip/core/us_block.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

using vsip::impl::Length;
using vsip::impl::extent;



/***********************************************************************
  Definitions
***********************************************************************/


template <typename T>
inline T
identity(
  Length<1> /*extent*/,
  Index<1> idx,
  int      k)
{
  return static_cast<T>(k*idx[0] + 1);
}


template <typename T>
inline T
identity(
  Length<2> extent,
  Index<2>  idx,
  int       k)
{
  Index<2> offset;
  index_type i = (idx[0]+offset[0])*extent[1] + (idx[1]+offset[1]);
  return static_cast<T>(k*i+1);
}


template <dimension_type Dim,
	  typename       Block>
void
fill_block(Block& blk, int k)
{
  typedef typename Block::value_type value_type;

  Length<Dim> ex = extent<Dim>(blk);
  for (Index<Dim> idx; valid(ex,idx); next(ex, idx))
  {
    put(blk, idx, identity<value_type>(ex, idx, k));
  }
}


template <dimension_type Dim,
	  typename       Block>
void
check_block(Block& blk, int k)
{
  typedef typename Block::value_type value_type;

  Length<Dim> ex = extent<Dim>(blk);
  for (Index<Dim> idx; valid(ex,idx); next(ex, idx))
  {
    test_assert(equal( get(blk, idx),
		  identity<value_type>(ex, idx, k)));
  }
}




template <dimension_type Dim,
	  typename       BlockT>
void
test(Domain<Dim> const& dom)
{
  using vsip::impl::conditional;
  using vsip::impl::is_same;
  using vsip::impl::Aligned_allocator;
  using vsip::interleaved_complex;
  using vsip::impl::scalar_of;

  typedef typename BlockT::value_type                       value_type;
  static storage_format_type const storage_format = get_block_layout<BlockT>::storage_format;
  typedef impl::Storage<storage_format, value_type>           storage_type;
  typedef typename storage_type::type                       ptr_type;

  typedef typename
    conditional<storage_format == interleaved_complex,
            Aligned_allocator<value_type>,
            Aligned_allocator<typename scalar_of<value_type>::type> >
      ::type alloc_type;
  alloc_type alloc;

  ptr_type ptr = storage_type::allocate(alloc, dom.size());

  {
    BlockT block(dom, ptr);

    assert(block.size() == dom.size());
    for (dimension_type d=0; d<Dim; ++d)
      assert(block.size(Dim, d) == dom[d].size());

    fill_block<Dim>(block, 5);
    check_block<Dim>(block, 5);
  }

  storage_type::deallocate(alloc, ptr, dom.size());
}



int
main()
{
  using vsip::impl::Us_block;
  using vsip::Layout;
  using vsip::dense;
  using vsip::aligned;
  using vsip::interleaved_complex;
  using vsip::split_complex;

  typedef Layout<1, row1_type, dense, interleaved_complex> LP_ri1;
  typedef Layout<2, row2_type, dense, interleaved_complex> LP_ri2;
  typedef Layout<2, row2_type, dense, split_complex> LP_rs2;

  test<1, Us_block<1, float, LP_ri1, Local_map> >(Domain<1>(5));

  test<2, Us_block<2, float, LP_ri2, Local_map> >(Domain<2>(5, 7));
  test<2, Us_block<2, float, LP_rs2, Local_map> >(Domain<2>(5, 7));
}
