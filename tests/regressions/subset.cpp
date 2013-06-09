//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Test subset of a subset.

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/parallel.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename M, typename T>
void
test_subset()
{

  typedef Dense<1, T, row1_type, M> block_type;
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

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // [1] Subset of a Subset_map was erroneously using the parent map
  //     for the subset.

  test_subset<Local_map, float>(); // OK
#if OVXX_PARALLEL
  test_subset<Map<>, float>(); // [1]
  test_subset<Replicated_map<1>, float>(); // OK
#endif
}
