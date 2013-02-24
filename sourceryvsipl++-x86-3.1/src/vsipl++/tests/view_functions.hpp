/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/view_functions.hpp
    @author  Stefan Seefeld
    @date    2005-03-16
    @brief   VSIPL++ Library: Unit tests for View expressions.

    This file contains unit tests for View expressions.
*/

#ifndef VSIP_TESTS_VIEW_FUNCTIONS_HPP
#define VSIP_TESTS_VIEW_FUNCTIONS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>
#include <complex>
#include <iostream>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

// Unary func(View) call.
#define TEST_UNARY(func, type, value)		   \
  {                                                \
    Vector<type, Dense<1, type> > v1(3, value);    \
    Vector<type, Dense<1, type> > v2 = func(v1);   \
    test_assert(equal(v2.get(0), func(v1.get(0))));     \
  }

// Unary func(View) call.
#define TEST_UNARY_RETN(func, type, retn, value)   \
  {                                                \
    Vector<type, Dense<1, type> > v1(3, value);    \
    Vector<type, Dense<1, retn> > v2 = func(v1);   \
    test_assert(equal(v2.get(0), func(v1.get(0))));     \
  }

// Binary func(View, View) call.
#define TEST_VV(func, type, value1, value2)		  \
  {                                                       \
    Vector<type, Dense<1, type> > v1(2, value1);          \
    Vector<type, Dense<1, type> > v2(2, value2);          \
    Vector<type, Dense<1, type> > v3 = func(v1, v2);      \
    test_assert(equal(v3.get(0), func(v1.get(0), v2.get(0)))); \
  }

// Binary func(View, View) call.
#define TEST_VV_RETN(func, type, retn, value1, value2)    \
  {                                                       \
    Vector<type, Dense<1, type> > v1(2, value1);          \
    Vector<type, Dense<1, type> > v2(2, value2);          \
    Vector<retn, Dense<1, retn> > v3 = func(v1, v2);      \
    test_assert(equal(v3.get(0), func(v1.get(0), v2.get(0)))); \
  }

// Binary func(View, Scalar) call.
#define TEST_VS(func, type, value1, value2)               \
  {                                                       \
    Vector<type, Dense<1, type> > v1(2, value1);          \
    Vector<type, Dense<1, type> > v2 = func(v1, value2);  \
    test_assert(equal(v2.get(0), func(v1.get(0), value2)));    \
  }

// Binary func(View, Scalar) call.
#define TEST_VS_RETN(func, type, retn, value1, value2)	  \
  {                                                       \
    Vector<type, Dense<1, type> > v1(2, value1);          \
    Vector<retn, Dense<1, retn> > v2 = func(v1, value2);  \
    test_assert(equal(v2.get(0), func(v1.get(0), value2)));    \
  }

// Binary func(Scalar, View) call.
#define TEST_SV(func, type, value1, value2)               \
  {                                                       \
    Vector<type, Dense<1, type> > v1(2, value1);          \
    Vector<type, Dense<1, type> > v2 = func(value2, v1);  \
    test_assert(equal(v2.get(0), func(value2, v1.get(0))));    \
  }

// Binary func(Scalar, View) call.
#define TEST_SV_RETN(func, type, retn, value1, value2)	  \
  {                                                       \
    Vector<type, Dense<1, type> > v1(2, value1);          \
    Vector<retn, Dense<1, retn> > v2 = func(value2, v1);  \
    test_assert(equal(v2.get(0), func(value2, v1.get(0))));    \
  }

// Ternary func(View, View, View) call.
#define TEST_VVV(func, type, value1, value2, value3)	             \
  {                                                                  \
    Vector<type, Dense<1, type> > v1(2, value1);                     \
    Vector<type, Dense<1, type> > v2(2, value2);                     \
    Vector<type, Dense<1, type> > v3(2, value3);                     \
    Vector<type, Dense<1, type> > v4 = func(v1, v2, v3);             \
    test_assert(equal(v4.get(0), func(v1.get(0), v2.get(0), v3.get(0)))); \
    test_assert(equal(v4.get(1), func(v1.get(1), v2.get(1), v3.get(1)))); \
  }

// Ternary func(Scalar, View, View) call.
#define TEST_SVV(func, type, value1, value2, value3)                 \
  {                                                                  \
    type scalar = value1;                                            \
    Vector<type, Dense<1, type> > v1(2, value2);                     \
    Vector<type, Dense<1, type> > v2(2, value3);                     \
    Vector<type, Dense<1, type> > v3 = func(scalar, v1, v2);         \
    test_assert(equal(v3.get(0), func(scalar, v1.get(0), v2.get(0))));    \
    test_assert(equal(v3.get(1), func(scalar, v1.get(1), v2.get(1))));    \
  }

// Ternary func(View, Scalar, View) call.
#define TEST_VSV(func, type, value1, value2, value3)                 \
  {                                                                  \
    type scalar = value1;                                            \
    Vector<type, Dense<1, type> > v1(2, value2);                     \
    Vector<type, Dense<1, type> > v2(2, value3);                     \
    Vector<type, Dense<1, type> > v3 = func(v1, scalar, v2);         \
    test_assert(equal(v3.get(0), func(v1.get(0), scalar, v2.get(0))));    \
    test_assert(equal(v3.get(1), func(v1.get(1), scalar, v2.get(1))));    \
  }

// Ternary func(View, View, Scalar) call.
#define TEST_VVS(func, type, value1, value2, value3)                 \
  {                                                                  \
    type scalar = value1;                                            \
    Vector<type, Dense<1, type> > v1(2, value2);                     \
    Vector<type, Dense<1, type> > v2(2, value3);                     \
    Vector<type, Dense<1, type> > v3 = func(v1, v2, scalar);         \
    test_assert(equal(v3.get(0), func(v1.get(0), v2.get(0), scalar)));    \
    test_assert(equal(v3.get(1), func(v1.get(1), v2.get(1), scalar)));    \
  }

// Ternary func(View, Scalar, Scalar) call.
#define TEST_VSS(func, type, value1, value2, value3)                 \
  {                                                                  \
    type scalar1 = value1;                                           \
    type scalar2 = value2;                                           \
    Vector<type, Dense<1, type> > v(2, value3);                      \
    Vector<type, Dense<1, type> > v2 = func(v, scalar1, scalar2);    \
    test_assert(equal(v2.get(0), func(v.get(0), scalar1, scalar2)));      \
    test_assert(equal(v2.get(1), func(v.get(1), scalar1, scalar2)));      \
  }

// Ternary func(Scalar, View, Scalar) call.
#define TEST_SVS(func, type, value1, value2, value3)                 \
  {                                                                  \
    type scalar1 = value1;                                           \
    type scalar2 = value2;                                           \
    Vector<type, Dense<1, type> > v(2, value3);                      \
    Vector<type, Dense<1, type> > v2 = func(scalar1, v, scalar2);    \
    test_assert(equal(v2.get(0), func(scalar1, v.get(0), scalar2)));      \
    test_assert(equal(v2.get(1), func(scalar1, v.get(1), scalar2)));      \
  }

// Ternary func(Scalar, Scalar, View) call.
#define TEST_SSV(func, type, value1, value2, value3)                 \
  {                                                                  \
    type scalar1 = value1;                                           \
    type scalar2 = value2;                                           \
    Vector<type, Dense<1, type> > v(2, value3);                      \
    Vector<type, Dense<1, type> > v2 = func(scalar1, scalar2, v);    \
    test_assert(equal(v2.get(0), func(scalar1, scalar2, v.get(0))));      \
    test_assert(equal(v2.get(1), func(scalar1, scalar2, v.get(1))));      \
  }

#define TEST_BINARY(name, type, value1, value2) \
TEST_VV(name, type, value1, value2)             \
TEST_VS(name, type, value1, value2)             \
TEST_SV(name, type, value1, value2)

#define TEST_BINARY_RETN(name, type, retn, value1, value2) \
TEST_VV_RETN(name, type, retn, value1, value2)		   \
TEST_VS_RETN(name, type, retn, value1, value2)		   \
TEST_SV_RETN(name, type, retn, value1, value2)

#define TEST_TERNARY(name, type, value1, value2, value3) \
TEST_VVV(name, type, value1, value2, value3)             \
TEST_SVV(name, type, value1, value2, value3)             \
TEST_VSV(name, type, value1, value2, value3)             \
TEST_VVS(name, type, value1, value2, value3)             \
TEST_VSS(name, type, value1, value2, value3)             \
TEST_SVS(name, type, value1, value2, value3)             \
TEST_SSV(name, type, value1, value2, value3)

#endif // VSIP_TESTS_VIEW_FUNCTIONS_HPP
