/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regressions/localview_of_slice.cpp
    @author  Jules Bergmann
    @date    2006-03-24
    @brief   VSIPL++ Library: Regression tests for local view of
             distributed sliced view.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/map.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

// Test that local view of a distributed matrix are empty on
// processors with no local subblock.
//
// 060324:
//  - Works correctly, test included for comparison.

void
test_localview()
{
  typedef float                            T;
  typedef Map<>                            map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;

  length_type np   = num_processors();
  length_type rows = (np > 1) ? (np - 1) : 1;
  length_type cols = 8;

  map_type map(rows, 1);
  Matrix<T, block_type> view(rows, cols, T(), map);


  // If the local processor has a subblock, it should have exactly one row.
  // Otherwise, it should have no rows.

  length_type local_rows = (subblock(view) != no_subblock) ? 1 : 0;

  test_assert(view.local().size(0) == local_rows);

#if VERBOSE
  cout << local_processor() << ": "
       << "size: " << view.local().size(0) << "  "
       << "lrows: " << local_rows << "  ";
  if (subblock(view) == no_subblock)
    cout << "sb: no_subblock";
  else
    cout << "sb: " << subblock(view);
  cout << endl;
#endif
}



// Test that local view of a slice of a distributed matrix are empty on
// processors with no local subblock.
// 
// 060324
//  - This test requires num_processors() > 1 to trigger error condition.
//  - Error condition fixed.

void
test_localview_of_slice()
{
  typedef float                            T;
  typedef Map<>                            map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;

  length_type np   = num_processors();
  length_type rows = np + 1;
  length_type cols = 8;

  map_type map(np, 1);
  Matrix<T, block_type> view(rows, cols, T(), map);

  for (index_type r = 0; r < rows; ++r)
  {
    length_type is_local = (subblock(view.row(r)) != no_subblock) ? 1 : 0;

#if VERBOSE
    cout << local_processor() << ": "
	 << view.row(r).local().size() << ", "
	 << is_local
	 << endl;
#endif

    test_assert(view.row(r).local().size() == is_local * cols);
  }
}



// Test that local view of a subset of a distributed matrix are empty on
// processors with no local subblock.
// 
// 060324
//  - This test does not trigger an error condition, since subviews 
//    are not currently allowed to break up distributed dimensions.
//    Attempting to run this test will throw 'unimplemented'.

void
test_localview_of_subset()
{
  typedef float                            T;
  typedef Map<>                            map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;

  length_type np   = num_processors();
  length_type rows = np + 1;
  length_type cols = 8;

  map_type map(np, 1);
  Matrix<T, block_type> view(rows, cols, T(), map);

  for (index_type r = 0; r < rows; ++r)
  {
    Domain<2> dom(Domain<1>(r, 1, 1),
		  Domain<1>(cols));
    length_type is_local = (subblock(view(dom)) != no_subblock) ? 1 : 0;

#if VERBOSE
    cout << local_processor() << ": "
	 << view(dom).local().size() << ", "
	 << is_local
	 << endl;
#endif

    test_assert(view(dom).local().size() == is_local * cols);
  }
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_localview();
  test_localview_of_slice();

  // See function comment.
  // test_localview_of_subset();
}
