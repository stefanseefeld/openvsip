/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/reductions-bool.cpp
    @author  Jules Bergmann
    @date    2005-07-11
    @brief   VSIPL++ Library: Tests for math boolean reductions.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>

using namespace vsip;
using namespace vsip_csl;


void
simple_tests()
{
  Vector<bool> bvec(4, true);

  test_assert(alltrue(bvec) == true);
  test_assert(anytrue(bvec) == true);

  bvec(2) = false;

  test_assert(alltrue(bvec) == false);
  test_assert(anytrue(bvec) == true);

  bvec(0) = false;
  bvec(1) = false;
  bvec(3) = false;

  test_assert(alltrue(bvec) == false);
  test_assert(anytrue(bvec) == false);

  Vector<int> vec(3);

  vec(0) = 0x00ff;
  vec(1) = 0x119f;
  vec(2) = 0x92f7;

  test_assert(alltrue(vec) == 0x0097);
  test_assert(anytrue(vec) == 0x93ff);
}



/***********************************************************************
  bool anytrue, alltrue tests.
***********************************************************************/

template <typename       StoreT,
	  dimension_type Dim>
void
test_bool(Domain<Dim> const& dom)
{
  typedef typename StoreT::value_type T;

  StoreT      true_store (dom, true);
  StoreT      false_store(dom, false);
  length_type size = true_store.view.size();

  test_assert(alltrue(true_store.view) == true);
  test_assert(anytrue(true_store.view) == true);

  test_assert(alltrue(false_store.view) == false);
  test_assert(anytrue(false_store.view) == false);

  put_nth(true_store.view, size-1, false);
  put_nth(false_store.view, size-1, true);

  test_assert(alltrue(true_store.view) == false);
  test_assert(anytrue(true_store.view) == true);

  test_assert(alltrue(false_store.view) == false);
  test_assert(anytrue(false_store.view) == true);
}



void
cover_bool()
{
  typedef bool T;

  test_bool<Storage<1, T> >(Domain<1>(15));
  
  test_bool<Storage<2, T, row2_type> >(Domain<2>(15, 17));
  test_bool<Storage<2, T, col2_type> >(Domain<2>(15, 17));
  
  test_bool<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7));
  test_bool<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7));
  test_bool<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7));
  test_bool<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7));
  test_bool<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7));
  test_bool<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
   
  simple_tests();

  cover_bool();
}
