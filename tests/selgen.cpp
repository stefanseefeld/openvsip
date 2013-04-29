//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>
#include <complex>
#include <vsip/initfin.hpp>
#include <vsip/selgen.hpp>
#include <functional>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;
using vsip_csl::view_equal;

void 
test_first()
{
  Vector<float> v1 = ramp(3.f, 2.f, 3);
  Vector<float> v2 = ramp(0.f, 3.f, 3);
  index_type i = first(0, std::less<float>(), v1, v2);
  test_assert(equal(i, static_cast<index_type>(3)));
  i = first(5, std::less<float>(), v1, v2);
  test_assert(equal(i, static_cast<index_type>(5)));
}

void 
test_indexbool()
{
  Vector<bool> v(5, false);
  v.put(0, true);
  v.put(2, true);
  v.put(4, true);
  Vector<Index<1> > indices1(5);
  length_type length = indexbool(v, indices1);
  test_assert(length == 3 && 
	 indices1.get(0) == Index<1>(0) && 
	 indices1.get(1) == Index<1>(2) &&
	 indices1.get(2) == Index<1>(4));

  Matrix<bool> m(5, 5, false);
  m.put(0, 2, true);
  m.put(2, 3, true);
  m.put(4, 2, true);
  Vector<Index<2> > indices2(5);
  length = indexbool(m, indices2);
  test_assert(length == 3 && 
	 indices2.get(0) == Index<2>(0, 2) && 
	 indices2.get(1) == Index<2>(2, 3) &&
	 indices2.get(2) == Index<2>(4, 2));
}

void test_gather_scatter()
{
  Matrix<float> m(5, 5, 0.);
  Vector<Index<2> > indices(4);
  indices.put(0, Index<2>(0, 0));
  indices.put(1, Index<2>(1, 3));
  indices.put(2, Index<2>(3, 2));
  indices.put(3, Index<2>(4, 1));
  put(m, indices.get(0), 1.f);
  put(m, indices.get(1), 2.f);
  put(m, indices.get(2), 3.f);
  put(m, indices.get(3), 4.f);

  Vector<float> v = gather(m, indices);
  test_assert(equal(v.get(0), 1.f));
  test_assert(equal(v.get(1), 2.f));
  test_assert(equal(v.get(2), 3.f));
  test_assert(equal(v.get(3), 4.f));

  Matrix<float> m2(5, 5, 0.);
  scatter(v, indices, m2);
  test_assert(view_equal(m, m2));
}

void
test_clip()
{
  Vector<float> v = ramp(0.f, 1.f, 5);
  Vector<float> result = clip(v, 1.1f, 2.9f, 1.1f, 2.9f);
  test_assert(equal(result.get(0), 1.1f));
  test_assert(equal(result.get(1), 1.1f));
  test_assert(equal(result.get(2), 2.f));
  test_assert(equal(result.get(3), 2.9f));
  test_assert(equal(result.get(4), 2.9f));
}

void
test_invclip()
{
  Vector<float> v = ramp(0.f, 1.f, 5);
  Vector<float> result = invclip(v, 1.1f, 2.1f, 3.1f, 1.1f, 3.1f);
  test_assert(equal(result.get(0), 0.f));
  test_assert(equal(result.get(1), 1.f));
  test_assert(equal(result.get(2), 1.1f));
  test_assert(equal(result.get(3), 3.1f));
  test_assert(equal(result.get(4), 4.f));
}

void
test_swap()
{
  Matrix<float> m1(5, 5, 0.f);
  Matrix<float> m2(5, 5, 2.f);

  Matrix<float> t1(5, 5);
  t1 = m1;
  Matrix<float> t2(5, 5);
  t2 = m2;
  vsip::swap(t1, t2);
  test_assert(view_equal(t1, m2));
  test_assert(view_equal(t2, m1));
}

int 
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_first();
  test_indexbool();
  test_gather_scatter();
  test_clip();
  test_invclip();
  test_swap();
}
