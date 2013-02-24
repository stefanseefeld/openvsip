/* Copyright (c) 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/regr_complex_proxy.hpp
    @author  Jules Bergmann
    @date    2006-02-02
    @brief   VSIPL++ Library: Regression test for lvalue proxy objects
	                      representing complex<T> values.
*/

// Background: The following types of expression are failing.
//
//   complex<T>        a;
//   Lvalue_proxy<...> p; // where value is complex<T>
//   a = a + p;
//
// This is due to the definition of operator+ for complex:
//
//  template<typename _Tp>
//    inline complex<_Tp>
//    operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
//
// Template parameter deduction for _Tp fails when this function
// is considered for 'a + p'.  Since no type for _Tp exists that
// makes the parameterized 'const complex<_Tp>&' indentical to
// the argument type 'Lvalue_proxy<...>', and none of the allowable
// conversions apply, deduction for _Tp fails.  And since each
// argument-parameter pair is considered independently, with the
// process succeeding only if all conclusions are the same,
// deduction fails for the function.
//
// The regression illustrates this:
//  - Functions taking a non-template complex<float> (function_value
//    and function_cref) are considered (correctly) when given an old
//    Lvalue_proxy argument.
//  - However, functions taking a templated complex<float>
//    (function_template_value and function_template_cref) are not
//    considered for an old Lvalue_proxy argument.
//
// The fix is to have Lvalue_proxy derive from the complex<T> class when
// the proxy is for a complex<T> value.  Template deduction for _Tp
// then succeeds because complex<_Tp> is a base class of the argument
// type, which is an allowable argument conversion.
//
// For more detail, see Vandevoorde & Josuttis, secions 11.1 and 11.4.

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/core/lvalue_proxy.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/metaprogramming.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

void
function_value(complex<float> value)
{
  assert(value == complex<float>(1.0, -1.0));
}

void
function_cref(complex<float> const& value)
{
  assert(value == complex<float>(1.0, -1.0));
}

template <typename T>
void
function_template_value(complex<T> value)
{
  assert(value == complex<T>(1.0, -1.0));
}

template <typename T>
void
function_template_cref(complex<T> const& value)
{
  assert(value == complex<T>(1.0, -1.0));
}



void
test_fun()
{
  length_type size = 3;

  Dense<1, complex<float> > d(Domain<1>(size), complex<float> (42));

  impl::Lvalue_proxy<complex<float> , Dense<1, complex<float> >, 1> p(d, 1);

  complex<float> a = complex<float>(1.0, -1.0);

  function_value(a);
  function_cref(a);
  function_template_value(a);
  function_template_cref(a);

  p = complex<float>(1.0, -1.0);

  function_value(p);
  function_cref(p);
  function_template_value(p);
  function_template_cref(p);
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_fun();

  return 0;
}
