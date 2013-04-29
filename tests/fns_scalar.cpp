/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/fns_scalar.cpp
    @author  Jules Bergmann
    @date    2005-06-18
    @brief   VSIPL++ Library: Tests for scalar and elementwise functions
             from [math.fns.scalar].
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

using namespace std;
using namespace vsip;
using vsip_csl::equal;

template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_magsq()
{
  Storage<Dim, T1> Z;
  Storage<Dim, T2> A;

  put_nth(A.view, 0, T2(-3));

  Z.view = vsip::magsq(A.view);

  test_assert(equal(get_nth(Z.view, 0), T1(3*3)));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_mag()
{
  Storage<Dim, T1> Z;
  Storage<Dim, T2> A;

  put_nth(A.view, 0, T2(-3));

  Z.view = vsip::mag(A.view);

  test_assert(equal(get_nth(Z.view, 0), T1(3)));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       T3>
void
test_maxmgsq()
{
  Storage<Dim, T1> Z;
  Storage<Dim, T2> A;
  Storage<Dim, T3> B;

  put_nth(A.view, 0, T2(-3));
  put_nth(B.view, 0, T3(-4));

  Z.view = vsip::maxmgsq(A.view, B.view);

  test_assert(equal(get_nth(Z.view, 0), T1(4*4)));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2,
	  typename       T3>
void
test_minmgsq()
{
  Storage<Dim, T1> Z;
  Storage<Dim, T2> A;
  Storage<Dim, T3> B;

  put_nth(A.view, 0, T2(-3));
  put_nth(B.view, 0, T3(-4));

  Z.view = vsip::minmgsq(A.view, B.view);

  test_assert(equal(get_nth(Z.view, 0), T1(3*3)));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_arg()
{
  Storage<Dim, T1> Z;
  Storage<Dim, T2> A;

  T2 input(3., 6.);

  put_nth(A.view, 0, input);

  Z.view = vsip::arg(A.view);

  test_assert(equal(get_nth(Z.view, 0), std::arg(input)));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  
  // magsq
  test_magsq<0, float, complex<float> >();
  test_magsq<1, float, complex<float> >();
  test_magsq<2, float, complex<float> >();

  test_magsq<0, float, float>();
  test_magsq<1, float, float>();
  test_magsq<2, float, float>();

  // mag
  test_mag<0, float, complex<float> >();
  test_mag<1, float, complex<float> >();
  test_mag<2, float, complex<float> >();

  test_mag<0, float, float>();
  test_mag<1, float, float>();
  test_mag<2, float, float>();

  // maxmgsq
  test_maxmgsq<0, float, complex<float>, complex<float> >();
  test_maxmgsq<1, float, complex<float>, complex<float> >();
  test_maxmgsq<2, float, complex<float>, complex<float> >();

  test_maxmgsq<0, double, complex<double>, complex<double> >();
  test_maxmgsq<1, double, complex<double>, complex<double> >();
  test_maxmgsq<2, double, complex<double>, complex<double> >();

  test_maxmgsq<0, double, complex<float>, complex<double> >();
  test_maxmgsq<1, double, complex<float>, complex<double> >();
  test_maxmgsq<2, double, complex<float>, complex<double> >();

  test_maxmgsq<0, float, float, float>();
  test_maxmgsq<1, float, float, float>();
  test_maxmgsq<2, float, float, float>();

  test_maxmgsq<0, double, float, double>();
  test_maxmgsq<1, double, float, double>();
  test_maxmgsq<2, double, float, double>();

  // minmgsq
  test_minmgsq<0, float, complex<float>, complex<float> >();
  test_minmgsq<1, float, complex<float>, complex<float> >();
  test_minmgsq<2, float, complex<float>, complex<float> >();

  test_minmgsq<0, double, complex<double>, complex<double> >();
  test_minmgsq<1, double, complex<double>, complex<double> >();
  test_minmgsq<2, double, complex<double>, complex<double> >();

  test_minmgsq<0, double, complex<float>, complex<double> >();
  test_minmgsq<1, double, complex<float>, complex<double> >();
  test_minmgsq<2, double, complex<float>, complex<double> >();

  test_minmgsq<0, float, float, float>();
  test_minmgsq<1, float, float, float>();
  test_minmgsq<2, float, float, float>();

  test_minmgsq<0, double, float, double>();
  test_minmgsq<1, double, float, double>();
  test_minmgsq<2, double, float, double>();

  // arg (float = arg(float) not allowed.)
  test_arg<0, float, complex<float> >();
  test_arg<1, float, complex<float> >();
  test_arg<2, float, complex<float> >();
}
