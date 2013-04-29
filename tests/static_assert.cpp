/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/static_assert.cpp
    @author  Jules Bergmann
    @date    2005-02-08
    @brief   VSIPL++ Library: Unit tests for static_assert.hpp classes.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/static_assert.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip_csl;


/***********************************************************************
  Macros
***********************************************************************/

// This test contains negative-compile test cases.  To test, enable
// one of the tests manually and check that the compilation fails.

#define ILLEGAL1 0
#define ILLEGAL2 0
#define ILLEGAL3 0
#define ILLEGAL4 0
#define ILLEGAL5 0



/***********************************************************************
  Declaratins
***********************************************************************/


template <typename T>
class Test_assert_unsigned
  : vsip::impl::Assert_unsigned<T>
{
  T a;
};



/***********************************************************************
  Definitions
***********************************************************************/

void 
test_static_assert()
{
  using vsip::scalar_f;
  using vsip::cscalar_f;
  using std::complex;

  VSIP_IMPL_STATIC_ASSERT(true);
  VSIP_IMPL_STATIC_ASSERT(sizeof(float)          == sizeof(scalar_f));
  VSIP_IMPL_STATIC_ASSERT(sizeof(complex<float>) == sizeof(cscalar_f));

#if ILLEGAL1
  VSIP_IMPL_STATIC_ASSERT(false);
#endif

#if ILLEGAL2
  VSIP_IMPL_STATIC_ASSERT(sizeof(scalar_f) == sizeof(cscalar_f));
#endif
}



void
test_assert_unsigned()
{
  Test_assert_unsigned<unsigned> t1;
  Test_assert_unsigned<vsip::index_type> t2;
  Test_assert_unsigned<vsip::length_type> t3;

#if ILLEGAL3
  typedef int T;
#else
  typedef unsigned T;
#endif
  Test_assert_unsigned<T> t4;

#if ILLEGAL4
  Test_assert_unsigned<vsip::stride_type> t5;
  use_variable(t5);
#endif

#if ILLEGAL5
  Test_assert_unsigned<vsip::scalar_i> t5;
  use_variable(t5);
#endif

  use_variable(t1);
  use_variable(t2);
  use_variable(t3);
  use_variable(t4);
}



int
main()
{
  test_static_assert();
  test_assert_unsigned();
}
