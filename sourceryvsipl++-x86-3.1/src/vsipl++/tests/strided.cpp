/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description:
///   VSIPL++ Library: Unit tests for Strided.

#include <iostream>
#include <cassert>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/core/strided.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip_csl/test.hpp>

using namespace vsip;
using vsip_csl::equal;

using vsip::impl::Length;
using vsip::impl::extent;
using vsip::impl::valid;
using vsip::impl::next;

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
	  typename       Block>
void
test(Domain<Dim> const& dom)
{
  Block block(dom);

  fill_block<Dim>(block, 5);
  check_block<Dim>(block, 5);
}

template <typename T, pack_type P, storage_format_type C>
void
test_wrap()
{
  using vsip::Layout;

  Domain<1> dom1(15);
  test<1, vsip::impl::Strided<1, T, Layout<1, row1_type, P, C> > >(dom1);

  Domain<2> dom2(15, 17);
  test<2, vsip::impl::Strided<2, T, Layout<2, row2_type, P, C> > >(dom2);
  test<2, vsip::impl::Strided<2, T, Layout<2, col2_type, P, C> > >(dom2);
}


int
main(int argc, char **argv)
{
  using vsip::dense;
  using vsip::aligned;
  using vsip::interleaved_complex;
  using vsip::split_complex;

  vsip::vsipl library(argc, argv);

  test_wrap<float, dense, interleaved_complex>();
  test_wrap<float, aligned_16, interleaved_complex>();
  test_wrap<float, dense, split_complex>();
  test_wrap<float, aligned_16, split_complex>();

  test_wrap<complex<float>, dense, interleaved_complex>();
  test_wrap<complex<float>, aligned_16, interleaved_complex>();
  test_wrap<complex<float>, dense, split_complex>();
  test_wrap<complex<float>, aligned_16, split_complex>();
}
