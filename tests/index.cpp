//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <cassert>
#include <vsip/domain.hpp>
#include <test.hpp>

using namespace ovxx;

/// Test Index<1> interface conformance.
void
test_index_1()
{
  Index<1> a;
  test_assert(a[0] == 0);
  Index<1> b(1);
  a = b;
  test_assert(a == b);
  Index<1> c(2);
  test_assert(b != c);
  Index<1> d(c);
  test_assert(d == c);
}

/// Test Index<1> interface conformance.
void
test_index_2()
{
  Index<2> a;
  test_assert(a[0] == 0 && a[1] == 0);
  Index<2> b(0, 1);
  test_assert(b[0] == 0 && b[1] == 1);
  a = b;
  test_assert(a == b);
  Index<2> c(2, 2);
  test_assert(b != c);
  Index<2> d(c);
  test_assert(d == c);
}

/// Test Index<1> interface conformance.
void
test_index_3()
{
  Index<3> a;
  test_assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
  Index<3> b(0, 1, 2);
  test_assert(b[0] == 0 && b[1] == 1 && b[2] == 2);
  a = b;
  test_assert(a == b);
  Index<3> c(2, 2, 2);
  test_assert(b != c);
  Index<3> d(c);
  test_assert(d == c);
}

int
main()
{
  test_index_1();
  test_index_2();
  test_index_3();
}
