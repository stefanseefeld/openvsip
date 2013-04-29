/* Copyright (c) 2007 by CodeSourcery, LLC.  All rights reserved. */

/** @file    tests/vsip_csl/error_db.cpp
    @author  Jules Bergmann
    @date    2007-12-07
    @brief   VSIPL++ Library: Unit tests for error_db
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0
#define SAVE_IMAGES 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>

using namespace vsip;
using namespace vsip_csl;



/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T>
void
test_vector(int what, T max)
{
  length_type size = 48;

  Rand<T> r(1);

  Vector<T> v1(size);
  Vector<T> v2(size);

  double expect;
  double delta = 1e-5;

  switch(what)
  {
  case 0:
    v1 = r.randu(size);
    v2 = v1;
    expect = -201;
    break;
  case 1:
    v1 = 0; v1(0) = max;
    v2 = 0; v2(0) = max;
    expect = -201;
    break;
  case 2:
    // These values will overflow magsq() of unsigned char if it is not
    // cast by error_db.
    v1 = 0; v1(0) = max;
    v2 = 0; v2(0) = max-1;
    expect = -10 * log10(2.0*max*max) + delta;
    break;
  }

  double error = error_db(v1, v2);

#if VERBOSE
  std::cout << "error: " << error << "  expect: " << expect << std::endl;
#endif

  test_assert(error <= expect);
}



template <typename T>
void
test_vector_cases(T max)
{
  test_vector<float>(0, max);
  test_vector<float>(1, max);
  test_vector<float>(2, max);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_vector_cases<float>         (255);
  test_vector_cases<int>           (255);
  test_vector_cases<unsigned int>  (255);
  test_vector_cases<signed short>  (255);
  test_vector_cases<unsigned short>(255);
  test_vector_cases<signed char>   (127);
  test_vector_cases<unsigned char> (255);
}
