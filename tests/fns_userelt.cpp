/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/fns_userelt.cpp
    @author  Stefan Seefeld
    @date    2005-07-26
    @brief   VSIPL++ Library: Unit tests for [math.fns.userelt]

    This file contains unit tests for [math.fns.userelt].
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>
#include <complex>

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>

using namespace vsip;

typedef Vector<float, Dense<1, float> > DVector;
typedef Vector<int, Dense<1, int> > IVector;

int my_unary(float v) 
{
  return static_cast<int>(std::ceil(v));
}

int my_binary(float v1, float v2) 
{
  return static_cast<int>(v1 + v2);
}

int my_ternary(float v1, float v2, float v3) 
{ 
  return static_cast<int>(v1 + v2 + v3);
}

struct my_func_obj
{
  int operator() (float v) const 
  { return my_unary(v);}
  int operator() (float v1, float v2) const 
  { return my_binary(v1, v2);}
  int operator() (float v1, float v2, float v3) const 
  { return my_ternary(v1, v2, v3);}
};

/***********************************************************************
  unary functions
***********************************************************************/

void unary_funptr()
{
  DVector input(3, 1.5);
  DVector result = unary(my_unary, input);
  test_assert(result.get(0) == my_unary(input(0)));
  test_assert(result.get(1) == my_unary(input(1)));
  test_assert(result.get(2) == my_unary(input(2)));
}

void unary_stdfunc()
{
  DVector input(3, 1.5);
  DVector result = unary<int>(std::ptr_fun(my_unary), input);
  test_assert(result.get(0) == my_unary(input(0)));
  test_assert(result.get(1) == my_unary(input(1)));
  test_assert(result.get(2) == my_unary(input(2)));
}

void unary_func()
{
  DVector input(3, 1.5);
  DVector result = unary<int>(my_func_obj(), input);
  test_assert(result.get(0) == my_unary(input(0)));
  test_assert(result.get(1) == my_unary(input(1)));
  test_assert(result.get(2) == my_unary(input(2)));
}

/***********************************************************************
  binary functions
***********************************************************************/

void binary_funptr()
{
  DVector input1(3, 1.5);
  DVector input2(3, .6);
  DVector result = binary(my_binary, input1, input2);
  test_assert(result.get(0) == my_binary(input1(0), input2(0)));
  test_assert(result.get(1) == my_binary(input1(1), input2(1)));
  test_assert(result.get(2) == my_binary(input1(2), input2(2)));
}

void binary_stdfunc()
{
  DVector input1(3, 1.5);
  DVector input2(3, .6);
  DVector result = binary<int>(std::ptr_fun(my_binary), input1, input2);
  test_assert(result.get(0) == my_binary(input1(0), input2(0)));
  test_assert(result.get(1) == my_binary(input1(1), input2(1)));
  test_assert(result.get(2) == my_binary(input1(2), input2(2)));
}

void binary_func()
{
  DVector input1(3, 1.5);
  DVector input2(3, .6);
  DVector result = binary<int>(my_func_obj(), input1, input2);
  test_assert(result.get(0) == my_binary(input1(0), input2(0)));
  test_assert(result.get(1) == my_binary(input1(1), input2(1)));
  test_assert(result.get(2) == my_binary(input1(2), input2(2)));
}

/***********************************************************************
  ternary functions
***********************************************************************/

void ternary_funptr()
{
  DVector input1(3, 1.5);
  DVector input2(3, .6);
  DVector input3(3, .9);
  DVector result = ternary(my_ternary, input1, input2, input3);
  test_assert(result.get(0) == my_ternary(input1(0), input2(0), input3(0)));
  test_assert(result.get(1) == my_ternary(input1(1), input2(1), input3(1)));
  test_assert(result.get(2) == my_ternary(input1(2), input2(2), input3(2)));
}

void ternary_func()
{
  DVector input1(3, 1.5);
  DVector input2(3, .6);
  DVector input3(3, .9);
  DVector result = ternary<int>(my_func_obj(), input1, input2, input3);
  test_assert(result.get(0) == my_ternary(input1(0), input2(0), input3(0)));
  test_assert(result.get(1) == my_ternary(input1(1), input2(1), input3(1)));
  test_assert(result.get(2) == my_ternary(input1(2), input2(2), input3(2)));
}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  unary_funptr();
  unary_stdfunc();
  unary_func();

  binary_funptr();
  binary_stdfunc();
  binary_func();

  ternary_funptr();
  ternary_func();
}
