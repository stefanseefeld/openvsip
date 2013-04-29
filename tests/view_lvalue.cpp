//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

// View lvalue accessors (tests).

//    Tests of the scalar operator() on views, which provide lvalue access to
//    the view contents.  These are roughly the same tests that appear in
//    lvalue-proxy.cpp, but using the high-level view interfaces.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip_csl;


template <typename View>
static void
probe_vector (View v)
{
  v(1) = 4;
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) ==  4);
  test_assert (v.get(2) == 42);

  v(1) += 3;
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) ==  7);
  test_assert (v.get(2) == 42);

  v(1) -= 5;
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) ==  2);
  test_assert (v.get(2) == 42);

  v(1) *= 3;
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) ==  6);
  test_assert (v.get(2) == 42);

  v(1) /= 2;
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) ==  3);
  test_assert (v.get(2) == 42);

  (v(1) = 12) = 10;
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) == 10);
  test_assert (v.get(2) == 42);

  v(1) = v(0);
  test_assert (v.get(0) == 42);
  test_assert (v.get(1) == 42);
  test_assert (v.get(2) == 42);
}

template <typename View>
static void
probe_matrix(View m)
{
  m(0,1) = 4;
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) ==  4); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);

  m(0,1) += 3;
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) ==  7); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);

  m(0,1) -= 5;
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) ==  2); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);

  m(0,1) *= 3;
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) ==  6); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);

  m(0,1) /= 2;
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) ==  3); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);

  (m(0,1) = 12) = 10;
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) == 10); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);

  m(0,1) = m(0,0);
  test_assert(m.get(0,0) == 42); test_assert(m.get(0,1) == 42); test_assert(m.get(0,2) == 42);
  test_assert(m.get(1,0) == 42); test_assert(m.get(1,1) == 42); test_assert(m.get(1,2) == 42);
  test_assert(m.get(2,0) == 42); test_assert(m.get(2,1) == 42); test_assert(m.get(2,2) == 42);
}

template <typename View>
static void
probe_tensor(View t)
{
  t(0,1,2) = 4;
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)== 4);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);

  t(0,1,2) += 3;
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)== 7);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);

  t(0,1,2) -= 5;
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)== 2);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);

  t(0,1,2) *= 3;
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)== 6);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);

  t(0,1,2) /= 2;
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)== 3);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);

  (t(0,1,2) = 12) = 10;
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)==10);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);

  t(0,1,2) = t(0,0,0);
  test_assert(t.get(0,0,0)==42); test_assert(t.get(0,0,1)==42); test_assert(t.get(0,0,2)==42);
  test_assert(t.get(0,1,0)==42); test_assert(t.get(0,1,1)==42); test_assert(t.get(0,1,2)==42);
  test_assert(t.get(0,2,0)==42); test_assert(t.get(0,2,1)==42); test_assert(t.get(0,2,2)==42);
  test_assert(t.get(1,0,0)==42); test_assert(t.get(1,0,1)==42); test_assert(t.get(1,0,2)==42);
  test_assert(t.get(1,1,0)==42); test_assert(t.get(1,1,1)==42); test_assert(t.get(1,1,2)==42);
  test_assert(t.get(1,2,0)==42); test_assert(t.get(1,2,1)==42); test_assert(t.get(1,2,2)==42);
  test_assert(t.get(2,0,0)==42); test_assert(t.get(2,0,1)==42); test_assert(t.get(2,0,2)==42);
  test_assert(t.get(2,1,0)==42); test_assert(t.get(2,1,1)==42); test_assert(t.get(2,1,2)==42);
  test_assert(t.get(2,2,0)==42); test_assert(t.get(2,2,1)==42); test_assert(t.get(2,2,2)==42);
}


template <typename View>
static void
test_vector()
{
  View v(3);
  v = 42;
  probe_vector(v);

  View base(6);
  base = 99;
  View sub = base(vsip::Domain<1>(3)*2);
  sub = 42;
  probe_vector(sub);
}

template <typename View>
static void
test_matrix()
{
  View m(3,3);
  m = 42;
  probe_matrix(m);

  View base(6,6);
  base = 99;
  View sub = base(vsip::Domain<2>(3,3)*2);
  sub = 42;
  probe_matrix(sub);

  base = 99;
  base.row(0) = 42;
  probe_vector(base.row(0));
}

template <typename View>
static void
test_tensor()
{
  View t(3,3,3);
  t = 42;
  probe_tensor(t);
}

int
main(int argc, char** argv)
{
  using vsip::Vector;
  using vsip::Matrix;
  using vsip::Tensor;
  using vsip::Dense;
  using vsip::scalar_f;

  vsip::vsipl init(argc, argv);

  // Tests with Dense test true lvalue access.
  test_vector<Vector<scalar_f, Dense<1, scalar_f> > >();
  test_matrix<Matrix<scalar_f, Dense<2, scalar_f> > >();
  test_tensor<Tensor<scalar_f, Dense<3, scalar_f> > >();
}
