/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    complex.cpp
    @author  Jules Bergmann
    @date    2005-03-17
    @brief   VSIPL++ Library: Unit tests for [complex] items.

    This file has unit tests for functionality defined in the [complex]
    section of the VSIPL++ specification.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/complex.hpp>

#include <vsip_csl/test.hpp>

using vsip_csl::equal;
using vsip_csl::view_equal;

/***********************************************************************
  Macros
***********************************************************************/

#define TEST_COMPLEX_FUNCTIONS						\
  complex<float> a(3.f, -4.f);						\
  complex<float> b(1.f,  0.f);						\
  complex<float> z;							\
  int            i = 1;							\
  float          f = 1.f;						\
  float          s;							\
  bool           x;							\
									\
  z = cos(a);								\
  z = cosh(a);								\
  z = exp(a);								\
  z = log(a);								\
  z = log10(a);								\
  z = sin(a);								\
  z = sinh(a);								\
  z = sqrt(a);								\
  z = tan(a);								\
  z = tanh(a);								\
  z = conj(a); test_assert(equal(z, complex<float>(3.f, 4.f)));		\
  s = real(a); test_assert(equal(s,  3.f));				\
  s = imag(a); test_assert(equal(s, -4.f));				\
  s = abs(a);  test_assert(equal(s,  5.f));				\
  s = arg(a);								\
  s = norm(a); test_assert(equal(s,  25.f));				\
  z = pow(a, i); test_assert(equal(z, a));				\
  z = pow(a, b); test_assert(equal(z, a));				\
  z = pow(a, f); test_assert(equal(z, a));				\
  z = pow(f, b); test_assert(equal(z, b));				\
  z = operator+(a, b); test_assert(equal(z, complex<float>(4.f, -4.f)));\
  z = operator-(a, b); test_assert(equal(z, complex<float>(2.f, -4.f)));\
  z = operator*(a, b); test_assert(equal(z, complex<float>(3.f, -4.f)));\
  z = operator/(a, b); test_assert(equal(z, complex<float>(3.f, -4.f)));\
  x = operator==(a, b); test_assert(x == false);			\
  x = operator!=(a, b); test_assert(x == true);				\
									\
  z = vsip::cos(a);							\
  z = vsip::cosh(a);							\
  z = vsip::exp(a);							\
  z = vsip::log(a);							\
  z = vsip::log10(a);							\
  z = vsip::sin(a);							\
  z = vsip::sinh(a);							\
  z = vsip::sqrt(a);							\
  z = vsip::tan(a);							\
  z = vsip::tanh(a);							\
  z = vsip::conj(a); test_assert(equal(z, complex<float>(3.f, 4.f)));	\
  s = vsip::real(a); test_assert(equal(s,  3.f));			\
  s = vsip::imag(a); test_assert(equal(s, -4.f));			\
  s = vsip::abs(a);  test_assert(equal(s,  5.f));			\
  s = vsip::arg(a);							\
  s = vsip::norm(a); test_assert(equal(s, 25.f));			\
  z = vsip::pow(a, i); test_assert(equal(z, a));			\
  z = vsip::pow(a, b); test_assert(equal(z, a));			\
  z = vsip::pow(a, f); test_assert(equal(z, a));			\
  z = vsip::pow(f, b); test_assert(equal(z, b));			\
  z = vsip::operator+(a, b); test_assert(equal(z, complex<float>(4.f, -4.f))); \
  z = vsip::operator-(a, b); test_assert(equal(z, complex<float>(2.f, -4.f))); \
  z = vsip::operator*(a, b); test_assert(equal(z, complex<float>(3.f, -4.f))); \
  z = vsip::operator/(a, b); test_assert(equal(z, complex<float>(3.f, -4.f))); \
  x = vsip::operator==(a, b); test_assert(x == false);			\
  x = vsip::operator!=(a, b); test_assert(x == true);			\
									\
  z = std::cos(a);							\
  z = std::cosh(a);							\
  z = std::exp(a);							\
  z = std::log(a);							\
  z = std::log10(a);							\
  z = std::sin(a);							\
  z = std::sinh(a);							\
  z = std::sqrt(a);							\
  z = std::tan(a);							\
  z = std::tanh(a);							\
  z = std::conj(a);							\
  z = std::conj(a); test_assert(equal(z, complex<float>(3.f, 4.f)));	\
  s = std::real(a); test_assert(equal(s,  3.f));			\
  s = std::imag(a); test_assert(equal(s, -4.f));			\
  s = std::abs(a);  test_assert(equal(s,  5.f));			\
  s = std::arg(a);							\
  s = std::norm(a); test_assert(equal(s, 25.f));			\
  z = std::pow(a, i); test_assert(equal(z, a));				\
  z = std::pow(a, b); test_assert(equal(z, a));				\
  z = std::pow(a, f); test_assert(equal(z, a));				\
  z = std::pow(f, b); test_assert(equal(z, b));				\
  z = std::operator+(a, b); test_assert(equal(z, complex<float>(4.f, -4.f))); \
  z = std::operator-(a, b); test_assert(equal(z, complex<float>(2.f, -4.f))); \
  z = std::operator*(a, b); test_assert(equal(z, complex<float>(3.f, -4.f))); \
  z = std::operator/(a, b); test_assert(equal(z, complex<float>(3.f, -4.f))); \
  x = std::operator==(a, b); test_assert(x == false);			\
  x = std::operator!=(a, b); test_assert(x == true);				\
  /* last */



/***********************************************************************
  Definitions
***********************************************************************/

namespace test_using_both
{

using namespace vsip;
using namespace std;

void
test()
{
  TEST_COMPLEX_FUNCTIONS
}

} // namespace test_using_both



namespace test_using_vsip
{

using namespace vsip;

void
test()
{
  TEST_COMPLEX_FUNCTIONS
}

} // namespace test_using_vsip



namespace test_using_std
{

using namespace std;

void
test()
{
  TEST_COMPLEX_FUNCTIONS
}

} // namespace test_using_std



namespace test_using_none
{
using vsip::complex;

void
test()
{
  TEST_COMPLEX_FUNCTIONS
}

} // namespace test_using_none



template <typename T>
void
test_complex()
{
  using vsip::complex;

  T		real = T(1);
  T		imag = T(2);
  complex<T>	c1(real, imag);

  test_assert(equal(c1.real(), real));
  test_assert(equal(c1.imag(), imag));

  complex<T>	c2 = c1 + T(1);

  test_assert(equal(c2.real(), T(real + 1)));
  test_assert(equal(c2.imag(), imag));

  complex<T>	c3 = c1 + complex<T>(T(0), T(1));

  test_assert(equal(c3.real(), real));
  test_assert(equal(c3.imag(), T(imag + 1)));
}



/// Test polar conversion functions.

template <typename T>
void
test_polar()
{
  using vsip::complex;
  using vsip::recttopolar;
  using vsip::polartorect;

  complex<T>	c1;
  complex<T>	c2;

  T		mag;
  T		phase;
  T		pi = 4*std::atan(1.f);

  c1 = complex<T>(T(1), T(0));
  recttopolar(c1, mag, phase);
  test_assert(equal(mag,   T(1)));
  test_assert(equal(phase, T(0)));
  c2 = polartorect(mag, phase);
  test_assert(equal(c1, c2));

  c1 = complex<T>(T(0), T(1));
  recttopolar(c1, mag, phase);
  test_assert(equal(mag,   T(1)));
  test_assert(equal(phase, pi/2));
  c2 = polartorect(mag, phase);
  test_assert(equal(c1, c2));

  c1 = complex<T>(T(1), T(1));
  vsip::recttopolar(c1, mag, phase);
  test_assert(equal(mag,   std::sqrt(T(2))));
  test_assert(equal(phase, pi/4));
  c2 = polartorect(mag, phase);
  test_assert(equal(c1, c2));


  c2 = polartorect(T(3));
  test_assert(equal(c2, complex<T>(T(3), T(0))));
}

template <typename T>
void
test_polar_view(vsip::length_type size)
{
  using vsip::complex;
  using vsip::recttopolar;
  using vsip::polartorect;

  vsip::Vector<complex<T> > v1(size), v2(size);
  vsip::Vector<T> mag(size), phase(size);

  T pi = 4*std::atan(1.f);

  v1 = complex<T>(T(1), T(0));
  recttopolar(v1, mag, phase);
  test_assert(equal(mag(0),   T(1)));
  test_assert(equal(phase(0), T(0)));
  v2 = polartorect(mag, phase);
  test_assert(view_equal(v1, v2));

  v1 = complex<T>(T(0), T(1));
  recttopolar(v1, mag, phase);
  test_assert(equal(mag(0),   T(1)));
  test_assert(equal(phase(0), pi/2));
  v2 = polartorect(mag, phase);
  test_assert(view_equal(v1, v2));

  v1 = complex<T>(T(1), T(1));
  recttopolar(v1, mag, phase);
  test_assert(equal(mag(0),   std::sqrt(T(2))));
  test_assert(equal(phase(0), pi/4));
  v2 = polartorect(mag, phase);
  test_assert(view_equal(v1, v2));


  v2 = polartorect(T(3));
  test_assert(equal(v2(0), complex<T>(T(3), T(0))));
}

template <typename T1, typename T2>
inline void
test_cmplx(vsip::length_type size)
{
  vsip::Vector<T1> v1(size, 1.), v2(size, 2.);
  vsip::Vector<vsip::complex<T1> > c = vsip::cmplx(v1, v2);
  typename vsip::Vector<vsip::complex<T1> >::realview_type r = c.real();
  typename vsip::Vector<vsip::complex<T1> >::imagview_type i = c.imag();
  test_assert(view_equal(r, v1));
  test_assert(view_equal(i, v2));
}

/// Test that functions such as exp, cos, etc. are available for complex.

template <typename T>
void
test_exp(T x, T y)
{
  using vsip::complex;

  complex<T>	c1;
  complex<T>	c2;
  T		pi = 4*std::atan(1.f);

  c1 = complex<T>(x, y);
  c2 = exp(c1);

  test_assert(equal(c2.real(), T(exp(x)*cos(y))));
  test_assert(equal(c2.imag(), T(exp(x)*sin(y))));

  test_assert(equal(exp(complex<T>(0, pi)) + T(1),
	            complex<T>(0)));
}
  



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_using_both::test();
  test_using_vsip::test();
  test_using_std::test();
  test_using_none::test();

  test_complex<int>();
  test_complex<float>();
  test_complex<double>();
  test_complex<short>();

  test_polar<float>();
  test_polar<double>();

  test_polar_view<float>(5);
  test_polar_view<double>(6);

  test_cmplx<float, float>(2);
  test_cmplx<double, float>(3);
  test_cmplx<double, double>(4);

  test_exp<float> (1.f, 2.f);
  test_exp<double>(2.0, 1.0);
}
