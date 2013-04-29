//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Test Maxval of distributed expression.

#include <algorithm>
#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>

#include "test_common.hpp"

using namespace vsip;
using vsip_csl::equal;

template <typename MapT>
void
test_maxval()
{
  typedef float T;
  typedef Dense<1, T, row1_type, MapT> block_type;
  typedef Vector<T, block_type > view_type;

  MapT map;

  length_type size = 16;
  view_type   view(size, map);
  Index<1>    idx;
  int         k = 1;
  T           maxv;

  setup(view, 1);

  maxv = maxval(view, idx);

  test_assert(equal(maxv, T((size-1)*k)));
  test_assert(idx[0] == (size-1));


  maxv = maxval(magsq(view), idx);

  test_assert(equal(maxv, sq(T((size-1)*k))));
  test_assert(idx[0] == (size-1));
}



template <typename       T,
	  typename       MapT,
	  dimension_type Dim>
void
test_maxval_nd(Domain<Dim> const& dom)
{
  typedef typename Default_order<Dim>::type order_type;
  Storage<Dim, T, order_type, MapT> stor(dom, T(1));
  Index<Dim> idx;

  T mv = maxval(stor.view, idx);

  test_assert(mv == T(1));
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_maxval<Local_map>();
  test_maxval<Map<> >();
  test_maxval<Replicated_map<1> >();



  // A bug in generic_par_idx_op prevented index-reductions such as
  // maxval from working on distributed views with dimension greater
  // than 1.

  test_maxval_nd<float, Local_map>(Domain<1>(5));		// OK
  test_maxval_nd<float, Local_map>(Domain<2>(5, 8));		// OK
  test_maxval_nd<float, Local_map>(Domain<3>(4, 6, 8));		// OK

  test_maxval_nd<float, Map<> >(Domain<1>(5));			// OK
  test_maxval_nd<float, Map<> >(Domain<2>(5, 8));		// error
  test_maxval_nd<float, Map<> >(Domain<3>(4, 6, 8));		// error

  test_maxval_nd<float, Replicated_map<1> >(Domain<1>(5));	// OK
  test_maxval_nd<float, Replicated_map<2> >(Domain<2>(5, 8));	// error
  test_maxval_nd<float, Replicated_map<3> >(Domain<3>(4, 6, 8));// error

  return 0;
}
