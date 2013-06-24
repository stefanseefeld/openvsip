//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/domain.hpp>
#include <test.hpp>

using namespace ovxx;

/// Test Domain<1> interface conformance.
void
test_domain_1()
{
  Domain<1> d1;
  test_assert(d1.first() == 0 && d1.stride() == 1 && d1.length() == 1);
  d1 = Domain<1>(0, 1, 2);
  test_assert(d1.size() == d1.length());
  Domain<1> d2(1, -1, 2);
  test_assert(d2.first() == 1 && d2.stride() == -1 && d2.length() == 2);
  test_assert(d1 == d2);
  d1 = Domain<1>(100, 2, 2);
  test_assert(d1 != d2);
  d2 = d1;
  test_assert(d1.element_conformant(d2));
  d1.impl_add_in(10);
  test_assert(d1.element_conformant(d2));
  test_assert(d1 != d2);
  Domain<2> x;
  test_assert(!d1.product_conformant(x));
  // operator + (Domain<>, index_difference_type)
  d2 = d1 + 5;
  test_assert(d2.first() == d1.first() + 5);
  // operator + (index_difference_type, Domain<>)
  d2 = 5 + d2;
  test_assert(d2.first() == d1.first() + 10);
  // operator - (Domain<>, index_difference_type)
  d2 = d2 - 10;
  test_assert(d2.first() == d1.first());
  // operator * (Domain<>, stride_scalar_type)
  d2 = d2 * 5;
  test_assert(d2.stride() == d1.stride() * 5);
  // operator * (stride_scalar_type, Domain<>)
  d2 = 2 * d2;
  test_assert(d2.stride() == d1.stride() * 10);
  // operator / (Domain<>, stride_scalar_type)
  d2 = d2 / 10;
  test_assert(d2.stride() == d1.stride());
}

/// Test Domain<2> interface conformance.
void
test_domain_2()
{
  Domain<1> a(1, 1, 5);
  Domain<1> b(1, 1, 5);
  Domain<2> d1(a, b);
  test_assert(d1.size() == 25);
  Domain<2> d2(d1);
  test_assert (d1 == d2);
  Domain<2> d3;
  d3 = d1;
  test_assert (d1 == d3);
  test_assert(d1.element_conformant(d2));
  test_assert(d1.product_conformant(d2));
  d1.impl_add_in(10);
  test_assert(d1.element_conformant(d2));
  test_assert(d1 != d2);
  d3 = Domain<2>(a, Domain<1>(1, 1, 1));
  test_assert(!d3.product_conformant(d1));
  // operator + (Domain<>, index_difference_type)
  d2 = d1 + 5;
  test_assert(d2[0].first() == d1[0].first() + 5);
  test_assert(d2[1].first() == d1[1].first() + 5);
  // operator + (index_difference_type, Domain<>)
  d2 = 5 + d2;
  test_assert(d2[0].first() == d1[0].first() + 10);
  test_assert(d2[1].first() == d1[1].first() + 10);
  // operator - (Domain<>, index_difference_type)
  d2 = d2 - 10;
  test_assert(d2[0].first() == d1[0].first());
  test_assert(d2[1].first() == d1[1].first());
  // operator * (Domain<>, stride_scalar_type)
  d2 = d2 * 5;
  test_assert(d2[0].stride() == d1[0].stride() * 5);
  test_assert(d2[1].stride() == d1[1].stride() * 5);
  // operator * (stride_scalar_type, Domain<>)
  d2 = 2 * d2;
  test_assert(d2[0].stride() == d1[0].stride() * 10);
  test_assert(d2[1].stride() == d1[1].stride() * 10);
  // operator / (Domain<>, stride_scalar_type)
  d2 = d2 / 10;
  test_assert(d2[0].stride() == d1[0].stride());
  test_assert(d2[1].stride() == d1[1].stride());
}

/// Test Domain<3> interface conformance.
void
test_domain_3()
{
  Domain<1> a(1, 1, 5);
  Domain<1> b(1, 1, 5);
  Domain<1> c(1, 1, 5);
  Domain<3> d1(a, b, c);
  test_assert(d1.size() == 125);
  Domain<3> d2(d1);
  test_assert (d1 == d2);
  Domain<3> d3;
  d3 = d1;
  test_assert (d1 == d3);
  test_assert(d1.element_conformant(d2));
  d1.impl_add_in(10);
  test_assert(d1.element_conformant(d2));
  test_assert(d1 != d2);
  Domain<2> x;
  test_assert(!d1.product_conformant(x));
  // operator + (Domain<>, index_difference_type)
  d2 = d1 + 5;
  test_assert(d2[0].first() == d1[0].first() + 5);
  test_assert(d2[1].first() == d1[1].first() + 5);
  test_assert(d2[2].first() == d1[2].first() + 5);
  // operator + (index_difference_type, Domain<>)
  d2 = 5 + d2;
  test_assert(d2[0].first() == d1[0].first() + 10);
  test_assert(d2[1].first() == d1[1].first() + 10);
  test_assert(d2[2].first() == d1[2].first() + 10);
  // operator - (Domain<>, index_difference_type)
  d2 = d2 - 10;
  test_assert(d2[0].first() == d1[0].first());
  test_assert(d2[1].first() == d1[1].first());
  test_assert(d2[2].first() == d1[2].first());
  // operator * (Domain<>, stride_scalar_type)
  d2 = d2 * 5;
  test_assert(d2[0].stride() == d1[0].stride() * 5);
  test_assert(d2[1].stride() == d1[1].stride() * 5);
  test_assert(d2[2].stride() == d1[2].stride() * 5);
  // operator * (stride_scalar_type, Domain<>)
  d2 = 2 * d2;
  test_assert(d2[0].stride() == d1[0].stride() * 10);
  test_assert(d2[1].stride() == d1[1].stride() * 10);
  test_assert(d2[2].stride() == d1[2].stride() * 10);
  // operator / (Domain<>, stride_scalar_type)
  d2 = d2 / 10;
  test_assert(d2[0].stride() == d1[0].stride());
  test_assert(d2[1].stride() == d1[1].stride());
  test_assert(d2[2].stride() == d1[2].stride());
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_domain_1();
  test_domain_2();
  test_domain_3();
}
