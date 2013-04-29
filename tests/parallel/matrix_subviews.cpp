/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Unit tests for distributed matrix subviews.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/math.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>
#include "util.hpp"
#include "util-par.hpp"
#include "test_common.hpp"

using namespace vsip;
using namespace vsip_csl;



/***********************************************************************
  Test subviews of distributed matrix
***********************************************************************/

template <typename T,
	  typename MapT,
	  typename SubMapT>
void
test_matrix_subview(
  Domain<2> const& dom,
  Domain<2> const& sub_dom,
  MapT const&      map,
  SubMapT const&   sub_map)
{
  typedef Dense<2, T, row2_type, MapT>    block_t;
  typedef Matrix<T, block_t>              view_t;

  typedef Dense<2, T, row2_type, SubMapT> sub_block_t;
  typedef Matrix<T, sub_block_t>          sub_view_t;

  int k = 1;

  // Setup.
  view_t     view   (create_view<view_t>    (dom,     map));
  sub_view_t subview(create_view<sub_view_t>(sub_dom, sub_map));

  setup(view, k);

  // Take subview.
  subview = view(sub_dom);

  // Check.
  check(subview, k, sub_dom[0].first(), sub_dom[1].first());
}



void test1()
{
  length_type np, nr, nc;

  get_np_square(np, nr, nc);

  Map<> root_map;
  Map<> rc_map(nr, nc);
  Replicated_map<2> rep_map;

  Domain<2> dom(10, 10);
  Domain<2> sub_dom(Domain<1>(2, 1, 6), Domain<1>(4, 1, 5));

  // 061010: SV++ does not support matrix subviews across a
  // distributed dimension.  I.e. 'map' (the 1st map argument) must be
  // either mapped to a single processor, or replicated.

  test_matrix_subview<float>(dom, sub_dom, root_map, root_map);
  test_matrix_subview<float>(dom, sub_dom, root_map, rc_map);

  test_matrix_subview<float>(dom, sub_dom, rep_map, root_map);
  test_matrix_subview<float>(dom, sub_dom, rep_map, rc_map);
}



int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  test1();

  return 0;
}
