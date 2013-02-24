/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Test subset of a subset.

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using vsip_csl::equal;

template <typename MapT,
	  typename T>
void
test_subset()
{

  typedef Dense<1, T, row1_type, MapT> block_type;
  typedef Vector<T, block_type> view_type;

  typedef typename view_type::subview_type    subset1_type;
  typedef typename subset1_type::subview_type subset2_type;

  length_type size = 16;

  view_type view(size);
  for (index_type i=0; i<size; ++i)
    view.put(i, T(i));

  subset1_type sub1 = view(Domain<1>(4, 1, 8));
  subset2_type sub2 = sub1(Domain<1>(2, 1, 4));

  for (index_type i=0; i<4; ++i)
    test_assert(equal(sub2.get(i), T(4+2+i)));
}






/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // [1] Subset of a Subset_map was erroneously using the parent map
  //     for the subset.

  test_subset<Local_map, float>(); // OK
  test_subset<Map<>, float>(); // [1]
  test_subset<Replicated_map<1>, float>(); // OK

  return 0;
}
