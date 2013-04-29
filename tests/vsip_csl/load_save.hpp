/* Copyright (c) 2006-2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/vsip_csl/load_save.hpp
    @author  Jules Bergmann
    @date    2006-09-28
    @brief   VSIPL++ Library: Common code for load/save_view unit-tests.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/load_view.hpp>
#include <vsip_csl/save_view.hpp>

#include "util.hpp"
#include "test_common.hpp"

/***********************************************************************
  Definitions
***********************************************************************/

// Test a round-trip through a file:
//  - create data in view
//  - save to disk using 'save_view'
//  - load from disk using 'Load_view'
//  - check result.

template <typename       T,
	  typename       OrderT,
          vsip::dimension_type Dim,
	  typename       SaveMapT,
	  typename       LoadMapT>
void
test_ls(
  vsip::Domain<Dim> const& dom,
  SaveMapT const&    save_map,
  LoadMapT const&    load_map,
  int                k,
  bool               do_barrier = false,
  bool               swap_bytes = false)
{
  char const* filename = "test.load_view.tmpfile";

  typedef vsip::Dense<Dim, T, OrderT, SaveMapT> save_block_type;
  typedef typename vsip::impl::view_of<save_block_type>::type save_view_type;
  typedef vsip::Dense<Dim, T, OrderT, LoadMapT> load_block_type;
  typedef typename vsip::impl::view_of<load_block_type>::type load_view_type;

  save_view_type s_view(create_view<save_view_type>(dom, save_map));

  setup(s_view, k);

  // Because the same file is shared for all tests, Wait for any
  // processors still doing an earlier test.
  if (do_barrier) vsip::impl::default_communicator().barrier();

  vsip_csl::save_view(filename, s_view, swap_bytes);

  // Wait for all writers to complete before starting to read.
  if (do_barrier) vsip::impl::default_communicator().barrier();

  // Test load_view function.
  load_view_type l_view(create_view<load_view_type>(dom, load_map));
  vsip_csl::load_view(filename, l_view, swap_bytes);
  check(l_view, k);

  // Test Load_view class.
  vsip_csl::Load_view<Dim, T, OrderT, LoadMapT> 
    l_view_obj(filename, dom, load_map, swap_bytes);
  check(l_view_obj.view(), k);
}



template <typename T>
void
test_type()
{
  using namespace vsip;
  Local_map l_map;

  Map<> map_0(1, 1);				// Root map
  Map<> map_r(vsip::num_processors(), 1);
  Map<> map_c(1, vsip::num_processors());

  // Local_map tests
  if (vsip::local_processor() == 0)
  {
    test_ls<T, row1_type>      (Domain<1>(16),      l_map, l_map, 1, false);

    test_ls<T, row2_type>      (Domain<2>(7, 5),    l_map, l_map, 1, false);
    test_ls<T, col2_type>      (Domain<2>(4, 7),    l_map, l_map, 1, false);

    test_ls<T, row3_type>      (Domain<3>(5, 3, 6), l_map, l_map, 1, false);
    test_ls<T, col3_type>      (Domain<3>(4, 7, 3), l_map, l_map, 1, false);
    test_ls<T, tuple<0, 2, 1> >(Domain<3>(5, 3, 6), l_map, l_map, 1, false);
  }

  // Because the same file name is used for all invocations of test_ls,
  // it is possible that processors other than 0 can race ahead and
  // corrupt the file being used by processor 0.  To avoid this, we
  // use a barrier here.
  impl::default_communicator().barrier();

  // 1D tests
  test_ls<T, row1_type>      (Domain<1>(16),      map_0, map_0, 1, true);
  test_ls<T, row2_type>      (Domain<2>(7, 5),    map_0, map_0, 1, true);
  test_ls<T, col2_type>      (Domain<2>(4, 7),    map_0, map_0, 1, true);
  test_ls<T, row3_type>      (Domain<3>(5, 3, 6), map_0, map_0, 1, true);
  test_ls<T, tuple<1, 0, 2> >(Domain<3>(4, 7, 3), map_0, map_0, 1, true);
  test_ls<T, tuple<0, 2, 1> >(Domain<3>(5, 3, 6), map_0, map_0, 1, true);


  // 1D tests
  test_ls<T, row1_type>      (Domain<1>(16),      map_0, map_r, 1, true);

  // 2D tests
  test_ls<T, row2_type>      (Domain<2>(7, 5),    map_0, map_r, 1, true);
  test_ls<T, col2_type>      (Domain<2>(4, 7),    map_0, map_c, 1, true);

  // 3D tests
  test_ls<T, row3_type>      (Domain<3>(5, 3, 6), map_0, map_r, 1, true);
  test_ls<T, tuple<1, 0, 2> >(Domain<3>(4, 7, 3), map_0, map_c, 1, true);
  test_ls<T, tuple<0, 2, 1> >(Domain<3>(5, 3, 6), map_0, map_r, 1, true);

  // Big-endian tests
  test_ls<T, row1_type>      (Domain<1>(16),      map_0, map_0, 1, true, true);
  test_ls<T, row2_type>      (Domain<2>(7, 5),    map_0, map_r, 1, true, true);
  test_ls<T, row3_type>      (Domain<3>(5, 3, 6), map_0, map_r, 1, true, true);

  // As above, prevent processors from going on to the next set of
  // local tests before all the others are done reading.
  impl::default_communicator().barrier();
}


