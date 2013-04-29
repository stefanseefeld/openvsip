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
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

// Test whether the operator '+' can be instantiated
// for the given view types.
template <typename View1, typename View2>
void
add_instantiation_test(View1 const& v1, View2 const& v2)
{
  (void) (v1 + v2);
}

// Test whether the operator '+' yields the correct numeric value.
template <typename T1, typename T2>
void
add_numeric_test(T1 p, T2 q)
{
  typedef typename Promotion<T1, T2>::type T;
  typedef Vector<T, Dense<1, T> > Result;
  typedef Dense<1, T1> Block1;
  typedef Dense<1, T2> Block2;
  typedef const_Vector<T1, Block1> View1;
  typedef const_Vector<T2, Block2> View2;
  Domain<1> domain(1);
  Block1 b1(domain, p);
  Block2 b2(domain, q);
  Result result = View1(b1) + View2(b2);
  test_assert(equal(result.get(0), static_cast<T1>(p) + static_cast<T2>(q)));
}

template <typename View>
void
minus_instantiation_test(View const& v)
{
  (void) -v;
}

void
add_test()
{
  Vector<float, Dense<1, float> > v1(3, 10.);
  Vector<float, Dense<1, float> > v2(3, 10.);

  Vector<float, Dense<1, float> > v3 = 5.f + v1 + v2 + v2 + 5.f;
  test_assert(equal(v3.get(0), 5.f + v1.get(0) + v2.get(0) + v2.get(0) + 5.f));
  test_assert(equal(v3.get(1), 5.f + v1.get(1) + v2.get(1) + v2.get(1) + 5.f));
  test_assert(equal(v3.get(2), 5.f + v1.get(2) + v2.get(2) + v2.get(2) + 5.f));
}

void
sub_test()
{
  Vector<float, Dense<1, float> > v1(3, 10.);
  Vector<float, Dense<1, float> > v2(3, 10.);

  Vector<float, Dense<1, float> > v3 = 5.f - v1 - v2 - v2 - 5.f;
  test_assert(equal(v3.get(0), 5.f - v1.get(0) - v2.get(0) - v2.get(0) - 5.f));
  test_assert(equal(v3.get(1), 5.f - v1.get(1) - v2.get(1) - v2.get(1) - 5.f));
  test_assert(equal(v3.get(2), 5.f - v1.get(2) - v2.get(2) - v2.get(2) - 5.f));
}

void
mult_test()
{
  Vector<float, Dense<1, float> > v1(3, 10.);
  Vector<float, Dense<1, float> > v2(3, 10.);

  Vector<float, Dense<1, float> > v3 = 5.f * v1 * v2 * v2 * 5.f;
  test_assert(equal(v3.get(0), 5.f * v1.get(0) * v2.get(0) * v2.get(0) * 5.f));
  test_assert(equal(v3.get(1), 5.f * v1.get(1) * v2.get(1) * v2.get(1) * 5.f));
  test_assert(equal(v3.get(2), 5.f * v1.get(2) * v2.get(2) * v2.get(2) * 5.f));
}

void
div_test()
{
  Vector<float, Dense<1, float> > v1(3, 10.);
  Vector<float, Dense<1, float> > v2(3, 10.);

  Vector<float, Dense<1, float> > v3 = 5.f / v1 / v2 / v2 / 5.f;
  test_assert(equal(v3.get(0), 5.f / v1.get(0) / v2.get(0) / v2.get(0) / 5.f));
  test_assert(equal(v3.get(1), 5.f / v1.get(1) / v2.get(1) / v2.get(1) / 5.f));
  test_assert(equal(v3.get(2), 5.f / v1.get(2) / v2.get(2) / v2.get(2) / 5.f));
}

#define COMPARE_VV_TEST(v1, v2, op)                       \
{                                                         \
  Vector<bool, Dense<1, bool> > result = v1 op v2;        \
  for (length_type i = 0; i != result.length(); ++i)      \
    test_assert(equal(result.get(i), v1.get(i) op v2.get(i))); \
}

#define COMPARE_VS_TEST(v, s, op)                         \
{                                                         \
  Vector<bool, Dense<1, bool> > result = v op s;          \
  for (length_type i = 0; i != result.length(); ++i)      \
    test_assert(equal(result.get(i), v.get(i) op s));          \
}

#define COMPARE_SV_TEST(s, v, op)                         \
{                                                         \
  Vector<bool, Dense<1, bool> > result = s op v;          \
  for (length_type i = 0; i != result.length(); ++i)      \
    test_assert(equal(result.get(i), s op v.get(i)));          \
}

void comparison_test()
{
  Vector<int, Dense<1, int> > v1(3, 1);
  Vector<int, Dense<1, int> > v2(3, 1);

  // View op View
  COMPARE_VV_TEST(v1, v2, ==)
  COMPARE_VV_TEST(v1, v2, >=)
  COMPARE_VV_TEST(v1, v2, >)
  COMPARE_VV_TEST(v1, v2, <=)
  COMPARE_VV_TEST(v1, v2, <)
  COMPARE_VV_TEST(v1, v2, !=)
  COMPARE_VV_TEST(v1, v2, &&)
  // Not really a comparison, but the same signature, so...
  COMPARE_VV_TEST(v1, v2, ||)
  v2 = 2;
  COMPARE_VV_TEST(v1, v2, ==)
  COMPARE_VV_TEST(v1, v2, >=)
  COMPARE_VV_TEST(v1, v2, >)
  COMPARE_VV_TEST(v1, v2, <=)
  COMPARE_VV_TEST(v1, v2, <)
  COMPARE_VV_TEST(v1, v2, !=)
  COMPARE_VV_TEST(v1, v2, &&)
  COMPARE_VV_TEST(v1, v2, ||)

  // View op Scalar
  COMPARE_VS_TEST(v1, 1, ==)
  COMPARE_VS_TEST(v1, 1, >=)
  COMPARE_VS_TEST(v1, 1, >)
  COMPARE_VS_TEST(v1, 1, <=)
  COMPARE_VS_TEST(v1, 1, <)
  COMPARE_VS_TEST(v1, 1, !=)
  COMPARE_VS_TEST(v1, 1, &&)
  COMPARE_VS_TEST(v1, 1, ||)

  COMPARE_VS_TEST(v1, 3, ==)
  COMPARE_VS_TEST(v1, 3, >=)
  COMPARE_VS_TEST(v1, 3, >)
  COMPARE_VS_TEST(v1, 3, <=)
  COMPARE_VS_TEST(v1, 3, <)
  COMPARE_VS_TEST(v1, 3, !=)
  COMPARE_VS_TEST(v1, 3, &&)
  COMPARE_VS_TEST(v1, 3, ||)

  // Scalar op View
  COMPARE_SV_TEST(1, v1, ==)
  COMPARE_SV_TEST(1, v1, >=)
  COMPARE_SV_TEST(1, v1, >)
  COMPARE_SV_TEST(1, v1, <=)
  COMPARE_SV_TEST(1, v1, <)
  COMPARE_SV_TEST(1, v1, !=)
  COMPARE_SV_TEST(1, v1, &&)
  COMPARE_SV_TEST(1, v1, ||)

  COMPARE_SV_TEST(3, v1, ==)
  COMPARE_SV_TEST(3, v1, >=)
  COMPARE_SV_TEST(3, v1, >)
  COMPARE_SV_TEST(3, v1, <=)
  COMPARE_SV_TEST(3, v1, <)
  COMPARE_SV_TEST(3, v1, !=)
  COMPARE_SV_TEST(3, v1, &&)
  COMPARE_SV_TEST(3, v1, ||)
}

#undef COMPARE_VV_TEST
#undef COMPARE_VS_TEST
#undef COMPARE_SV_TEST

#define BINARY_OP_VV_TEST(v1, v2, op)                     \
{                                                         \
  Vector<int, Dense<1, int> > result = v1 op v2;          \
  for (length_type i = 0; i != result.length(); ++i)      \
    test_assert(equal(result.get(i), v1.get(i) op v2.get(i))); \
}

#define BINARY_OP_VS_TEST(v, s, op)                       \
{                                                         \
  Vector<int, Dense<1, int> > result = v op s;            \
  for (length_type i = 0; i != result.length(); ++i)      \
    test_assert(equal(result.get(i), v.get(i) op s));          \
}

#define BINARY_OP_SV_TEST(s, v, op)                       \
{                                                         \
  Vector<int, Dense<1, int> > result = s op v;            \
  for (length_type i = 0; i != result.length(); ++i)      \
    test_assert(equal(result.get(i), s op v.get(i)));          \
}

void
binary_op_test()
{
  Vector<int, Dense<1, int> > v1(3, 3);
  Vector<int, Dense<1, int> > v2(3, 7);

  BINARY_OP_VV_TEST(v1, v2, |)
  BINARY_OP_VV_TEST(v1, v2, &)
  BINARY_OP_VV_TEST(v1, v2, ^)

  BINARY_OP_VS_TEST(v1, 7, |)
  BINARY_OP_VS_TEST(v1, 7, &)
  BINARY_OP_VS_TEST(v1, 7, ^)

  BINARY_OP_SV_TEST(3, v1, |)
  BINARY_OP_SV_TEST(3, v1, &)
  BINARY_OP_SV_TEST(3, v1, ^)
}

#undef BINARY_OP_VV_TEST
#undef BINARY_OP_VS_TEST
#undef BINARY_OP_SV_TEST

void
binary_lxor_test()
{
  Vector<bool, Dense<1, bool> > v1(3, true);
  Vector<bool, Dense<1, bool> > v2(3, false);

  {
    Vector<bool, Dense<1, bool> > result = v1 ^ v2;
    for (length_type i = 0; i != result.length(); ++i)
      test_assert(equal(result.get(i), bool(v1.get(i) ^ v2.get(i))));
  }
  {
    bool const s = false;
    Vector<bool, Dense<1, bool> > result = v1 ^ s;
    for (length_type i = 0; i != result.length(); ++i)
      test_assert(equal(result.get(i), bool(v1.get(i) ^ s)));
  }
  {
    bool const s = true;
    Vector<bool, Dense<1, bool> > result = s ^ v1;
    for (length_type i = 0; i != result.length(); ++i)
      test_assert(equal(result.get(i), bool(s ^ v1.get(i))));
  }
}

void
subblock_test()
{
  using namespace vsip::impl;

  typedef Component_block<Dense<1, std::complex<float> >, Real_extractor> Real_block;
  typedef Component_block<Dense<1, std::complex<float> >, Imag_extractor> Imag_block;

  Vector<float, Dense<1, float> > v1(3, 10.);
  Dense<1, std::complex<float> > block(Domain<1>(3), 10.);

  Real_block real(block);
  Vector<float, Real_block> realview(real);
  Vector<float, Dense<1, float> > v2 = v1 + realview;
  test_assert(equal(v2.get(0), v1.get(0) + realview.get(0)));

  Imag_block imag(block);
  Vector<float, Imag_block> imagview(imag);
  Vector<float, Dense<1, float> > v3 = v1 + imagview;
  test_assert(equal(v3.get(0), v1.get(0) + imagview.get(0)));
}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  typedef Vector<float, Dense<1, float> > DVector;
  typedef const_Vector<float, Dense<1, float> > const_DVector;
  typedef Matrix<float, Dense<2, float> > DMatrix;
  typedef const_Matrix<float, Dense<2, float> > const_DMatrix;

  minus_instantiation_test(DVector(3, 10.));
  minus_instantiation_test(const_DVector(3, 10.));
  add_instantiation_test(DVector(3, 10.), DVector(3, 10.));
  add_instantiation_test(const_DVector(3, 10.), DVector(3, 10.));
  add_instantiation_test(DVector(3, 10.), const_DVector(3, 10.));
  add_instantiation_test(const_DVector(3, 10.), const_DVector(3, 10.));

  minus_instantiation_test(DMatrix(3, 3, 10.));
  minus_instantiation_test(const_DMatrix(3, 3, 10.));
  add_instantiation_test(DMatrix(3, 3, 10.), DMatrix(3, 3, 10.));
  add_instantiation_test(const_DMatrix(3, 3, 10.), DMatrix(3, 3, 10.));
  add_instantiation_test(DMatrix(3, 3, 10.), const_DMatrix(3, 3, 10.));
  add_instantiation_test(const_DMatrix(3, 3, 10.), const_DMatrix(3, 3, 10.));

#if 0 // test for expected compile error
  add_instantiation_test(DVector(3, 10.), DMatrix(3, 10.));
#endif

  // Simple test without any conversion: 0.6 + 0.6 == 1.2
  add_numeric_test<float, float>(0.6, 0.6);

  // Add one int view to one float view: (int)0.6 + 0.6 == 0.6
  add_numeric_test<int, float>(0.6, 0.6);

  add_test();
  sub_test();
  mult_test();
  div_test();
  comparison_test();
  binary_op_test();
  binary_lxor_test();

  subblock_test();
}
