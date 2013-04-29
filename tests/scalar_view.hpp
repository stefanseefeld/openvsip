//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

enum Op_type
{
  op_add,
  op_sub,
  op_mul,
  op_div
};

/// Utility class to hold an Op_type value as a distinct type.
template <Op_type Type> struct Op_holder {};


/// Trait to determine value type of a type.

/// For general types, type is value type.
template <typename T>
struct Value_type
{
  typedef T type;
};

/// For views, element type is value type.
template <typename T, typename Block>
struct Value_type<Vector<T, Block> >
{
  typedef T type;
};



/// Overload of test_case for add expression: res = a + b.

template <typename T1,
	  typename T2,
	  typename T3>
void
test_case(
  Op_holder<op_add>,
  T1 res,
  T2 a,
  T3 b)
{
  typedef typename Value_type<T1>::type value_type;

  res = a + b;
  
  for (index_type i=0; i<get_size(res); ++i)
  {
    test_assert(equal(get_nth(res, i),
		       value_type(get_nth(a, i) + get_nth(b, i))));
  }
}



/// Overload of test_case for subtract expression: res = a - b.

template <typename T1,
	  typename T2,
	  typename T3>
void
test_case(
  Op_holder<op_sub>,
  T1 res,
  T2 a,
  T3 b)
{
  typedef typename Value_type<T1>::type value_type;

  res = a - b;
  
  for (index_type i=0; i<get_size(res); ++i)
  {
    test_assert(equal(get_nth(res, i),
		       value_type(get_nth(a, i) - get_nth(b, i))));
  }
}



/// Overload of test_case for multiply expression: res = a * b.

template <typename T1,
	  typename T2,
	  typename T3>
void
test_case(
  Op_holder<op_mul>,
  T1 res,
  T2 a,
  T3 b)
{
  typedef typename Value_type<T1>::type value_type;

  res = a * b;
  
  for (index_type i=0; i<get_size(res); ++i)
  {
    test_assert(equal(get_nth(res, i),
		       value_type(get_nth(a, i) * get_nth(b, i))));
  }
}



/// Overload of test_case for divide expression: res = a / b.

template <typename T1,
	  typename T2,
	  typename T3>
void
test_case(
  Op_holder<op_div>,
  T1 res,
  T2 a,
  T3 b)
{
  typedef typename Value_type<T1>::type value_type;

  res = a / b;
  
  for (index_type i=0; i<get_size(res); ++i)
  {
    test_assert(equal(get_nth(res, i),
		       value_type(get_nth(a, i) / get_nth(b, i))));
  }
}



// Test given expression with various combinations of scalar vs view
// operands and stride-1 vs stride-N operands.

template <Op_type  op,
	  typename T1,
	  typename T2,
	  typename T3>
void
test_type()
{
  length_type size = 8;

  Vector<T1> big_res(2 * size);
  Vector<T2> big_a(2 * size);
  Vector<T3> big_b(2 * size);

  Vector<T1> res(size);
  Vector<T2> a(size);
  Vector<T3> b(size);

  typename Vector<T1>::subview_type res2 = big_res(Domain<1>(0, 2, size));
  typename Vector<T2>::subview_type a2   = big_a(Domain<1>(0, 2, size));
  typename Vector<T3>::subview_type b2   = big_b(Domain<1>(0, 2, size));

  T2 alpha = T2(2);
  T3 beta  = T3(3);

  a  = ramp(T2(1), T2(1),  size);
  b  = ramp(T3(1), T3(-2), size);
  a2 = ramp(T2(1), T2(1),  size);
  b2 = ramp(T3(1), T3(-2), size);

  test_case(Op_holder<op>(), res, a, b);
  test_case(Op_holder<op>(), res, alpha, b);
  test_case(Op_holder<op>(), res, a, beta);

  test_case(Op_holder<op>(), res, a2, b);
  test_case(Op_holder<op>(), res, a, b2);
  test_case(Op_holder<op>(), res, alpha, b2);
  test_case(Op_holder<op>(), res, a2, beta);

  test_case(Op_holder<op>(), res2, a, b);
  test_case(Op_holder<op>(), res2, a2, b);
  test_case(Op_holder<op>(), res2, a, b2);
  test_case(Op_holder<op>(), res2, alpha, b);
  test_case(Op_holder<op>(), res2, a, beta);

  test_case(Op_holder<op>(), res2, a2, b2);
  test_case(Op_holder<op>(), res2, alpha, b2);
  test_case(Op_holder<op>(), res2, a2, beta);
}



// Test an operation for various types.

template <Op_type op>
void
test()
{
  test_type<op, short, short, short>();
  test_type<op, int, short, short>();
  test_type<op, int, int, short>();
  test_type<op, int, short, int>();
  test_type<op, int, int, int>();

  test_type<op, float, float, float>();
  test_type<op, float, double, float>();
  test_type<op, float, float, double>();

  test_type<op, double, double, double>();
  test_type<op, double, double, float>();
  test_type<op, double, float,  double>();
  test_type<op, double, float,  float>();

  test_type<op, complex<float>,         float,  complex<float> >();
  test_type<op, complex<float>, complex<float>,         float  >();
  test_type<op, complex<float>, complex<float>, complex<float> >();

  test_type<op, complex<double>,         double,  complex<double> >();
  test_type<op, complex<double>, complex<double>,         double  >();
  test_type<op, complex<double>, complex<double>, complex<double> >();

}



template <Op_type op>
void
test_lite()
{
  test_type<op, complex<float>,         float,  complex<float> >();
  test_type<op, complex<float>, complex<float>,         float  >();
  test_type<op, complex<float>, complex<float>, complex<float> >();
}
