//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/domain.hpp>
#include <ovxx/domain_utils.hpp>
#include <test.hpp>

using namespace ovxx;

void
test_intersect()
{
  Domain<1> intr;

  {
    Domain<1> dom1(0, 1, 10);
    Domain<1> dom2(5, 1, 7);
    test_assert(intersect(dom1, dom2, intr));
    test_assert(intr.first()  == 5);
    test_assert(intr.stride() == 1);
    test_assert(intr.size()   == 5);
  }

  {
    Domain<1> dom1(0, 1, 10);
    Domain<1> dom2(10, 1, 8);
    test_assert(!intersect(dom1, dom2, intr));
  }

  // test dom2 non-unit-stride cases
  {
    Domain<1> dom1(0, 1, 10);
    Domain<1> dom2(5, 2, 10);
    test_assert(intersect(dom1, dom2, intr));
    test_assert(intr.first()  == 5);
    test_assert(intr.stride() == 2);
    test_assert(intr.size()   == 3);
  }

  { // Have to adjust first2 forward (fractional stride)
    Domain<1> dom1(5, 1, 10); // [   5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    Domain<1> dom2(4, 2, 3);  // [4,    6,    8]
    test_assert(intersect(dom1, dom2, intr));
    test_assert(intr.first()  == 6);
    test_assert(intr.stride() == 2);
    test_assert(intr.size()   == 2);
  }
  { // Have to adjust last2 backwards
    Domain<1> dom1(5, 1, 5);  // [   5, 6, 7, 8, 9]
    Domain<1> dom2(4, 2, 4);  // [4,    6,    8,    10]
    test_assert(intersect(dom1, dom2, intr));
    test_assert(intr.first()  == 6);
    test_assert(intr.stride() == 2);
    test_assert(intr.size()   == 2);
  }

  { // Have to adjust first2 forward (whole stride)
    Domain<1> dom1(4, 1, 4);  // (pg_dom)            [4, 5, 6, 7]
    Domain<1> dom2(0, 2, 4);  // (dom)   [0,    2,    4,    6]
    test_assert(intersect(dom1, dom2, intr));
    test_assert(intr.first()  == 4);
    test_assert(intr.stride() == 2);
    test_assert(intr.size()   == 2);
  }
}



void
test_subset_from_intr()
{
  {
    Domain<1> dom (0, 1, 4); // [0, 1, 2, 3]
    Domain<1> intr(2, 1, 2); // [      2, 3]
    Domain<1> sub = subset_from_intr(dom, intr);
    test_assert(sub.first()  == 2);
    test_assert(sub.stride() == 1);
    test_assert(sub.size()   == 2);
  }

  {
    Domain<1> dom (0, 2, 4); // [0,    2,    4,    6]
    Domain<1> intr(4, 2, 2); // [            4,    6]
    Domain<1> sub = subset_from_intr(dom, intr);
    test_assert(sub.first()  == 2);
    test_assert(sub.stride() == 1);
    test_assert(sub.size()   == 2);
  }
}



void
test_apply_intr()
{
  {
    Domain<1> x   (0, 2, 4);                // [0, 2, 4, 6]
    Domain<1> y   (0, 1, 4);                // [0, 1, 2, 3]
    Domain<1> intr(2, 1, 2);                // [      2, 3]
    Domain<1> app = apply_intr(x, y, intr); // [      4, 6]
    test_assert(app.first()  == 4);
    test_assert(app.stride() == 2);
    test_assert(app.size()   == 2);
  }
}


/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_intersect();
  test_subset_from_intr();
  test_apply_intr();
}
