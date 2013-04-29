/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/lvalue_proxy.hpp
    @author  Zack Weinberg
    @date    2005-05-04
    @brief   VSIPL++ Library: Lvalue proxy objects (tests).

    Explicit instantiation of lvalue proxy objects and tests of their
    functionality.  */

#include <vsip/core/lvalue_proxy.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/dense.hpp>
#include <vsip/initfin.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;


template <typename T>
struct test_traits
{ static T value() { return T(1); } };

template <typename T>
struct test_traits<complex<T> >
{ static complex<T> value() { return complex<T>(1, -1); } };

template <typename T>
void
test_proxy(void)
{
  length_type size = 3;

  Dense<1, T> d(Domain<1>(size), T(42));

  vsip::impl::Lvalue_proxy<T, Dense<1, T>, 1> p(d, 1);
  vsip::impl::Lvalue_proxy<T, Dense<1, T>, 1> q(d, 2);

  test_assert (p == T(42));

  T a;
  T b = T(4);
  T c;

  p = T(3);
  c = p;

  assert(c == T(3));
  assert(p == T(3));
  assert(d.get(1) == T(3));

  a = b + p;

  assert(a == T(7));

  a += p;

  assert(a == T(10));

  p += a;

  assert(p        == T(13));
  assert(d.get(1) == T(13));

  q = T(5);
  assert(q        == T(5));
  assert(d.get(2) == T(5));

  p *= q;

  assert(p        == T(5*13));
  assert(d.get(1) == T(5*13));

  q = p;

  assert(q        == T(5*13));
  assert(d.get(2) == T(5*13));

  assert(p == q);

  a = test_traits<T>::value();

  p = a;

  assert(p        == a);
  assert(d.get(1) == a);

  q = vsip_csl::impl_conj<T>(p);
  // q = vsip_csl::impl_conj(p);

  assert(q        == vsip_csl::impl_conj(a));
  assert(d.get(2) == vsip_csl::impl_conj(a));

  typedef typename vsip::impl::scalar_of<T>::type scalar_type;

  scalar_type x = scalar_type(1);

  a = T(0);
  p = T(0);
  q = T(0);

  a += x;
  p += x;

  q += scalar_type(1);
  q += 1;

  assert(a        == T(1));
  assert(p        == T(1));
  assert(d.get(1) == T(1));

  assert(q        == T(2));
  assert(d.get(2) == T(2));

  assert(a        == scalar_type(1));
  assert(p        == scalar_type(1));

#if 0
  // These are valid for T = float and complex<float>:
  assert(a        == 1.f);
  assert(p        == 1.f);

  // These are not valid:
  // NOT VALID: // assert(a        == 1);
  // NOT VALID: // assert(p        == 1);
#endif
}



template <typename T>
void
test_complex_proxy(void)
{
  length_type size = 3;

  Dense<1, complex<T> > d(Domain<1>(size), complex<T> (42));

  vsip::impl::Lvalue_proxy<complex<T> , Dense<1, complex<T> >, 1> p(d, 1);

  test_assert (p == complex<T> (42));

  complex<T> a;

  a = complex<T>(1, -1);
  p = complex<T>(1, -1);

  assert(a.real() == T(+1));
  assert(a.imag() == T(-1));

  assert(p.real() == T(+1));
  assert(p.imag() == T(-1));
  assert(d.get(1).real() == T(+1));
  assert(d.get(1).imag() == T(-1));

  p += a;

  assert(p == complex<T>(+2, -2));
  assert(p.real() == T(+2));
  assert(p.imag() == T(-2));
  assert(d.get(1) == complex<T>(+2, -2));
  assert(d.get(1).real() == T(+2));
  assert(d.get(1).imag() == T(-2));
}



template <typename T>
static void
test_1d (void)
{
  Dense<1, T> d(Domain<1>(3), 42);
  vsip::impl::Lvalue_proxy<T, Dense<1, T>, 1> p(d, 1);
  test_assert (p == T(42));

  p = 4;
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T( 4));
  test_assert (d.get(2) == T(42));

  p += 3;
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T( 7));
  test_assert (d.get(2) == T(42));

  p -= 5;
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T( 2));
  test_assert (d.get(2) == T(42));

  p *= 3;
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T( 6));
  test_assert (d.get(2) == T(42));

  p /= 2;
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T( 3));
  test_assert (d.get(2) == T(42));

  (p = 12) = 10;
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T(10));
  test_assert (d.get(2) == T(42));

  p = vsip::impl::Lvalue_proxy<T, Dense<1, T>, 1>(d, 0);
  test_assert (d.get(0) == T(42));
  test_assert (d.get(1) == T(42));
  test_assert (d.get(2) == T(42));
}

static void
test_2d (void)
{
  Dense<2> d(Domain<2>(3, 3), 42);
  vsip::impl::Lvalue_proxy<float, Dense<2>, 2> p(d, Index<2>(0, 1));
  test_assert (p == 42);

  p = 4;
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) ==  4); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);

  p += 3;
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) ==  7); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);

  p -= 5;
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) ==  2); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);

  p *= 3;
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) ==  6); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);

  p /= 2;
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) ==  3); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);

  (p = 12) = 10;
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) == 10); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);

  p = vsip::impl::Lvalue_proxy<float, Dense<2>, 2>(d, Index<2>(0, 0));
  test_assert(d.get(0,0) == 42); test_assert(d.get(0,1) == 42); test_assert(d.get(0,2) == 42);
  test_assert(d.get(1,0) == 42); test_assert(d.get(1,1) == 42); test_assert(d.get(1,2) == 42);
  test_assert(d.get(2,0) == 42); test_assert(d.get(2,1) == 42); test_assert(d.get(2,2) == 42);
}

static void
test_3d (void)
{
  Dense<3> d(Domain<3>(3, 3, 3), 42);
  vsip::impl::Lvalue_proxy<float, Dense<3>, 3> p(d, Index<3>(0, 1, 2));
  test_assert (p == 42);

  p = 4;
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)== 4);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);

  p += 3;
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)== 7);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);

  p -= 5;
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)== 2);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);

  p *= 3;
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)== 6);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);

  p /= 2;
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)== 3);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);

  (p = 12) = 10;
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)==10);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);

  p = vsip::impl::Lvalue_proxy<float, Dense<3>, 3>(d, Index<3>(0, 0, 0));
  test_assert(d.get(0,0,0)==42); test_assert(d.get(0,0,1)==42); test_assert(d.get(0,0,2)==42);
  test_assert(d.get(0,1,0)==42); test_assert(d.get(0,1,1)==42); test_assert(d.get(0,1,2)==42);
  test_assert(d.get(0,2,0)==42); test_assert(d.get(0,2,1)==42); test_assert(d.get(0,2,2)==42);
  test_assert(d.get(1,0,0)==42); test_assert(d.get(1,0,1)==42); test_assert(d.get(1,0,2)==42);
  test_assert(d.get(1,1,0)==42); test_assert(d.get(1,1,1)==42); test_assert(d.get(1,1,2)==42);
  test_assert(d.get(1,2,0)==42); test_assert(d.get(1,2,1)==42); test_assert(d.get(1,2,2)==42);
  test_assert(d.get(2,0,0)==42); test_assert(d.get(2,0,1)==42); test_assert(d.get(2,0,2)==42);
  test_assert(d.get(2,1,0)==42); test_assert(d.get(2,1,1)==42); test_assert(d.get(2,1,2)==42);
  test_assert(d.get(2,2,0)==42); test_assert(d.get(2,2,1)==42); test_assert(d.get(2,2,2)==42);
}

// Pseudo-block for testing static type equalities that should hold for
// any block.
struct PseudoBlock
{
  static const int dim = 1;
  typedef int &reference_type;
  typedef int const &const_reference_type;
};


template <typename T>
void
test_dense_traits()
{
  using vsip::impl::is_same;
  // For all three dimensional specializations of Dense, it should know
  // that a true lvalue is available, unless block's storage is split
  // complex (that is, T is complex and storage_format is split_complex).
  VSIP_IMPL_STATIC_ASSERT((
    (is_same< typename vsip::impl::Lvalue_factory_type<Dense<1, T> >::type,
                      vsip::impl::True_lvalue_factory<Dense<1, T> >
    >::value == true) ^ vsip::impl::is_split_block<Dense<1, T> >::value));
  VSIP_IMPL_STATIC_ASSERT((
    (is_same< typename vsip::impl::Lvalue_factory_type<Dense<2, T> >::type,
                      vsip::impl::True_lvalue_factory<Dense<2, T> >
    >::value == true) ^ vsip::impl::is_split_block<Dense<2, T> >::value));
  VSIP_IMPL_STATIC_ASSERT((
    (is_same< typename vsip::impl::Lvalue_factory_type<Dense<3, T> >::type,
                      vsip::impl::True_lvalue_factory<Dense<3, T> >
    >::value == true) ^ vsip::impl::is_split_block<Dense<3, T> >::value));

}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);
#if 0
  // Type equalities valid for any block.
  VSIP_IMPL_STATIC_ASSERT((
    vsip::impl::is_same<vsip::impl::Proxy_lvalue_factory<PseudoBlock>::reference_type,
                      vsip::impl::Lvalue_proxy<int, PseudoBlock, 1>
    >::value == true));

  VSIP_IMPL_STATIC_ASSERT((
     vsip::impl::is_same<vsip::impl::True_lvalue_factory<PseudoBlock>::reference_type,
                      PseudoBlock::reference_type
    >::value == true));
#endif

  // Some static test_assertions about the traits class.
  // For the pseudo-block above, it should make the conservative assumption
  // that a proxy lvalue must be used.
  VSIP_IMPL_STATIC_ASSERT((
     vsip::impl::is_same<vsip::impl::Lvalue_factory_type<PseudoBlock>::type,
                     vsip::impl::Proxy_lvalue_factory<PseudoBlock>
    >::value == true));

  test_dense_traits<float>();
  test_dense_traits<complex<float> >();

  test_proxy<int>            ();
  test_proxy<float>          ();
  test_proxy<complex<float> >();
  test_proxy<double>          ();
  test_proxy<complex<double> >();

  test_complex_proxy<float> ();
  test_complex_proxy<double>();

  test_1d<int>();
  test_1d<float>();
  test_1d<complex<float> >();

  test_2d ();
  test_3d ();

  return 0;
}
