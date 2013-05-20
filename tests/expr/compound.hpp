//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Tests for compound expressions, common definitions

#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/selgen.hpp>
#include <vsip/signal.hpp>
#include <test.hpp>

#define DEBUG 0

using namespace ovxx;

template <typename V>
struct Ramp
{
  typedef typename V::value_type T;

  static void apply(V &v, T start, T delta)
  {
    Vector<T, typename V::block_type> rv(v.block());
    rv = ramp(start, delta, v.size());
  }
};

template <typename V>
struct Check
{
  typedef typename V::value_type T;
  typedef typename scalar_of<T>::type scalar_type;

  static void
  eval(V &v, index_type index, T value, index_type line_number)
  {
    Vector<T, typename V::block_type> rv(v.block());
    // Verify the result using a slightly lower threshold since 'equal(val, chk)'
    //  fails on CELL builds due to small numerical inconsistencies.
    scalar_type err_threshold = 1e-3;
#if DEBUG
    if (!almost_equal(rv.get(index), value, err_threshold))
      std::cout << "Error at index " << index << ", " 
                << v(index) << " != " << value << std::endl
                << "Go to line number " << line_number 
                << " in " << __FILE__ << std::endl;
#endif
    test_assert(almost_equal(rv.get(index), value, err_threshold));
  }
};

#define CHECK0(v, r)   Check<view_type>::eval(v, i0, r##0, __LINE__)
#define CHECK1(v, r)   Check<view_type>::eval(v, i1, r##1, __LINE__)
#define CHECK2(v, r)   Check<view_type>::eval(v, i2, r##2, __LINE__)
#define CHECK3(v, r)   Check<view_type>::eval(v, i3, r##3, __LINE__)

template <typename V>
struct Get
{
  typedef typename V::value_type T;

  static T apply(V &v, index_type index)
  {
    Vector<T, typename V::block_type> rv(v.block());
    return rv.get(index);
  }
};

#define GET0(v)  Get<view_type>::apply(v, i0)
#define GET1(v)  Get<view_type>::apply(v, i1)
#define GET2(v)  Get<view_type>::apply(v, i2)
#define GET3(v)  Get<view_type>::apply(v, i3)

template <typename V, dimension_type D = V::dim>
struct Subview_every_other;

template <typename V>
struct Subview_every_other<V, 1>
{
  static typename V::subview_type 
  apply(V v, bool odd)
  {
    int start = odd ? 1 : 0;
    Domain<1> dom(start, 2, v.size(0)/2);
    return v(dom);
  }
};

template <typename V>
struct Subview_every_other<V, 2>
{
  static typename V::subview_type 
  apply(V v, bool odd)
  {
    int start = odd ? 1 : 0;
    Domain<2> dom(v.size(0), Domain<1>(start, 2, v.size(1)/2));
    return v(dom);
  }
};

template <typename V>
struct Subview_every_other<V, 3>
{
  static typename V::subview_type 
  apply(V v, bool odd)
  {
    int start = odd ? 1 : 0;
    Domain<3> dom(v.size(0), v.size(1), Domain<1>(start, 2, v.size(2)/2));
    return v(dom);
  }
};

template <typename V>
static typename V::subview_type get_even_subview(V v)
{
  bool odd = false;
  return Subview_every_other<V>::apply(v, odd);
}

template <typename V>
static typename V::subview_type get_odd_subview(V v)
{
  bool odd = true;
  return Subview_every_other<V>::apply(v, odd);
}

/// Elementwise tests run through several compound expressions and then
/// compare results at select indices with individually computed values.
///
template <dimension_type D, typename T>
void
test_elementwise(Domain<D> dom)
{
  typedef typename vsip::Dense<D, T> block_type;
  typedef typename view_of<block_type>::type view_type;
  block_type a_blk(dom);
  block_type b_blk(dom);
  block_type c_blk(dom);
  block_type d_blk(dom);
  block_type z_blk(dom);
  view_type a(a_blk);
  view_type b(b_blk);
  view_type c(c_blk);
  view_type d(d_blk);
  view_type z(z_blk);
  T r0 = T();
  T r1 = T();
  T r2 = T();

  length_type size = 1;
  for (vsip::dimension_type i = 0; i < D; ++i)
    size *= dom[i].size();
  index_type i0 = 3;
  index_type i1 = size / 2 - 1;
  index_type i2 = size - 3;

#if DEBUG
  std::cout << __PRETTY_FUNCTION__
            << " size = " << size 
            << " dom = " << dom
            << std::endl;
#endif

  Ramp<view_type>::apply(a, T(1), T(1)/T(size));
  Ramp<view_type>::apply(b, T(2), T(1)/T(size));
  Ramp<view_type>::apply(c, T(3), T(1)/T(size));
  Ramp<view_type>::apply(d, T(4), T(1)/T(size));
  z = T();


  // unary(view)
  z = sin(a);
  r0 = sin(GET0(a));
  r1 = sin(GET1(a));
  r2 = sin(GET2(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // unary(unary(view))
  z = sin(cos(a));
  r0 = sin(cos(GET0(a)));
  r1 = sin(cos(GET1(a)));
  r2 = sin(cos(GET2(a)));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // unary(binary(view, view))
  z = sin(a + b);
  r0 = sin(GET0(a) + GET0(b));
  r1 = sin(GET1(a) + GET1(b));
  r2 = sin(GET2(a) + GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // unary(ternary(view, view, view))
  z = sin(ma(a, b, c));
  r0 = sin(ma(GET0(a), GET0(b), GET0(c)));
  r1 = sin(ma(GET1(a), GET1(b), GET1(c)));
  r2 = sin(ma(GET2(a), GET2(b), GET2(c)));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);



  // binary(view, view)
  z = a * b;
  r0 = GET0(a) * GET0(b);
  r1 = GET1(a) * GET1(b);
  r2 = GET2(a) * GET2(b);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(unary(view),unary(view))
  z = sin(a) + cos(b);
  r0 = sin(GET0(a)) + cos(GET0(b));
  r1 = sin(GET1(a)) + cos(GET1(b));
  r2 = sin(GET2(a)) + cos(GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(unary(view),binary(view, view))
  z = sin(a) + b * c;
  r0 = sin(GET0(a)) + GET0(b) * GET0(c);
  r1 = sin(GET1(a)) + GET1(b) * GET1(c);
  r2 = sin(GET2(a)) + GET2(b) * GET2(c);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(binary(view, view),binary(view, view))
  z = (a + b) + c * d;
  r0 = (GET0(a) + GET0(b)) + GET0(c) * GET0(d);
  r1 = (GET1(a) + GET1(b)) + GET1(c) * GET1(d);
  r2 = (GET2(a) + GET2(b)) + GET2(c) * GET2(d);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  
  // binary(unary(view),ternary(view, view, view))
  z = sin(a) * ma(b, c, d);
  r0 = sin(GET0(a)) * ma(GET0(b), GET0(c), GET0(d));
  r1 = sin(GET1(a)) * ma(GET1(b), GET1(c), GET1(d));
  r2 = sin(GET2(a)) * ma(GET2(b), GET2(c), GET2(d));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);



  // ternary(view, view, view)
  z = ma(a, b, c);
  r0 = ma(GET0(a), GET0(b), GET0(c));
  r1 = ma(GET1(a), GET1(b), GET1(c));
  r2 = ma(GET2(a), GET2(b), GET2(c));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // ternary(view, unary(view), binary(view))
  z = ma(a, sin(b), c * d);
  r0 = ma(GET0(a), sin(GET0(b)), GET0(c) * GET0(d));
  r1 = ma(GET1(a), sin(GET1(b)), GET1(c) * GET1(d));
  r2 = ma(GET2(a), sin(GET2(b)), GET2(c) * GET2(d));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);



  // Mixed scalar/view tests

  // binary(view, scalar)
  z = a * T(5);
  r0 = GET0(a) * T(5);
  r1 = GET1(a) * T(5);
  r2 = GET2(a) * T(5);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(scalar, view)
  z = T(5) + a;
  r0 = T(5) + GET0(a);
  r1 = T(5) + GET1(a);
  r2 = T(5) + GET2(a);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(unary(view),scalar)
  z = sin(a) * T(5);
  r0 = sin(GET0(a)) * T(5);
  r1 = sin(GET1(a)) * T(5);
  r2 = sin(GET2(a)) * T(5);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);



  // Aliasing checks

  // unary(view)
  z = a;
  z = sin(z);
  r0 = sin(GET0(a));
  r1 = sin(GET1(a));
  r2 = sin(GET2(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(view, view), arg1 aliased
  z = a;
  z = z * b;
  r0 = GET0(a) * GET0(b);
  r1 = GET1(a) * GET1(b);
  r2 = GET2(a) * GET2(b);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(view, view), arg2 aliased
  z = b;
  z = a * z;
  r0 = GET0(a) * GET0(b);
  r1 = GET1(a) * GET1(b);
  r2 = GET2(a) * GET2(b);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // binary(view, view), both args aliased
  z = b;
  z = z * z;
  r0 = GET0(b) * GET0(b);
  r1 = GET1(b) * GET1(b);
  r2 = GET2(b) * GET2(b);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // ternary(view, view, view), arg1 not aliased
  z = b;
  z = ma(a, z, z);
  r0 = ma(GET0(a), GET0(b), GET0(b));
  r1 = ma(GET1(a), GET1(b), GET1(b));
  r2 = ma(GET2(a), GET2(b), GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // ternary(view, view, view), arg2 not aliased
  z = a;
  z = ma(z, b, z);
  r0 = ma(GET0(a), GET0(b), GET0(a));
  r1 = ma(GET1(a), GET1(b), GET1(a));
  r2 = ma(GET2(a), GET2(b), GET2(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // ternary(view, view, view), arg3 not aliased
  z = a;
  z = ma(z, z, c);
  r0 = ma(GET0(a), GET0(a), GET0(c));
  r1 = ma(GET1(a), GET1(a), GET1(c));
  r2 = ma(GET2(a), GET2(a), GET2(c));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);

  // ternary(view, view, view), all args aliased
  z = a;
  z = ma(z, z, z);
  r0 = ma(GET0(a), GET0(a), GET0(a));
  r1 = ma(GET1(a), GET1(a), GET1(a));
  r2 = ma(GET2(a), GET2(a), GET2(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
}



/// Elementwise subview tests are similar to the elementwise tests.
/// Using domains and/or other means to express subviews, we run through 
/// several compound expressions and then compare results at select indices 
/// with individually computed values.
///
template <dimension_type D, typename T>
void
test_elementwise_subview(Domain<D> dom)
{
  typedef typename vsip::Dense<D, T> block_type;
  typedef typename view_of<block_type>::type view_type;
  block_type a_blk(dom);
  block_type b_blk(dom);
  block_type c_blk(dom);
  block_type d_blk(dom);
  block_type z_blk(dom);
  view_type a(a_blk);
  view_type b(b_blk);
  view_type c(c_blk);
  view_type d(d_blk);
  view_type z(z_blk);
  T r0 = T();
  T r1 = T();
  T r2 = T();
  T r3 = T();

  length_type size = 1;
  for (vsip::dimension_type i = 0; i < D; ++i)
    size *= dom[i].size();
  index_type i0 = 2;
  index_type i1 = 3;
  index_type i2 = size - 4;
  index_type i3 = size - 3;

#if DEBUG
  std::cout << __PRETTY_FUNCTION__
            << " size = " << size 
            << " dom = " << dom
            << std::endl;
#endif

  Ramp<view_type>::apply(a, T(1), T(1)/T(size));
  Ramp<view_type>::apply(b, T(2), T(1)/T(size));
  Ramp<view_type>::apply(c, T(3), T(1)/T(size));
  Ramp<view_type>::apply(d, T(4), T(1)/T(size));
  z = T();

  
  // unary(subview)
  //  verify odd values are changed and even ones unaffected
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = sin(get_odd_subview(a));
  r1 = sin(GET1(a));
  r3 = sin(GET3(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  verify even values are changed and odd ones unaffected
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = sin(get_even_subview(a));
  r0 = sin(GET0(a));
  r2 = sin(GET2(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);

  // unary(unary(subview))
  //  odd values
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = sin(sin(get_odd_subview(a)));
  r1 = sin(sin(GET1(a)));
  r3 = sin(sin(GET3(a)));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even values
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = sin(sin(get_even_subview(a)));
  r0 = sin(sin(GET0(a)));
  r2 = sin(sin(GET2(a)));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);

  // unary(binary(subview, subview))
  //  odd values
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = sin(get_odd_subview(a) * get_odd_subview(b));
  r1 = sin(GET1(a) * GET1(b));
  r3 = sin(GET3(a) * GET3(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even values
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = sin(get_even_subview(a) * get_even_subview(b));
  r0 = sin(GET0(a) * GET0(b));
  r2 = sin(GET2(a) * GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);



  // binary(subview, unary(subview))
  //  odd values
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = get_odd_subview(a) * sin(get_odd_subview(b));
  r1 = GET1(a) * sin(GET1(b));
  r3 = GET3(a) * sin(GET3(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even values
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = get_even_subview(a) * sin(get_even_subview(b));
  r0 = GET0(a) * sin(GET0(b));
  r2 = GET2(a) * sin(GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);

  // binary(unary(subview), subview)
  //  odd values
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = sin(get_odd_subview(a)) * get_odd_subview(b);
  r1 = sin(GET1(a)) * GET1(b);
  r3 = sin(GET3(a)) * GET3(b);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even values
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = sin(get_even_subview(a)) * get_even_subview(b);
  r0 = sin(GET0(a)) * GET0(b);
  r2 = sin(GET2(a)) * GET2(b);
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);

  // binary(unary(subview), unary(subview))
  //  odd values
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = sin(get_odd_subview(a)) + cos(get_odd_subview(b));
  r1 = sin(GET1(a)) + cos(GET1(b));
  r3 = sin(GET3(a)) + cos(GET3(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even values
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = sin(get_even_subview(a)) + cos(get_even_subview(b));
  r0 = sin(GET0(a)) + cos(GET0(b));
  r2 = sin(GET2(a)) + cos(GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);



  // ternary(unary(subview), binary(subview, unary(subview)), ternary(subview))
  //  odd values
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = ma(
    exp(get_odd_subview(a)),
    sin(get_odd_subview(b)) + cos(get_odd_subview(c)),
    ma(get_odd_subview(a), get_odd_subview(b), get_odd_subview(c))
  );
  r1 = ma(
    exp(GET1(a)), 
    sin(GET1(b)) + cos(GET1(c)),
    ma(GET1(a), GET1(b), GET1(c)));
  r3 = ma(
    exp(GET3(a)), 
    sin(GET3(b)) + cos(GET3(c)),
    ma(GET3(a), GET3(b), GET3(c)));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even values
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = ma(
    exp(get_even_subview(a)),
    sin(get_even_subview(b)) + cos(get_even_subview(c)),
    ma(get_even_subview(a), get_even_subview(b), get_even_subview(c))
  );
  r0 = ma(
    exp(GET0(a)), 
    sin(GET0(b)) + cos(GET0(c)),
    ma(GET0(a), GET0(b), GET0(c)));
  r2 = ma(
    exp(GET2(a)), 
    sin(GET2(b)) + cos(GET2(c)),
    ma(GET2(a), GET2(b), GET2(c)));

  get_even_subview(z) = sin(get_even_subview(a)) + cos(get_even_subview(b));
  r0 = sin(GET0(a)) + cos(GET0(b));
  r2 = sin(GET2(a)) + cos(GET2(b));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);


  // Mixed sub-view expressions

  // binary(unary(subview1), unary(subview2)), aliasing two inputs
  //  odd <-- odd, even
  Ramp<view_type>::apply(z, T(5), T(1)/T(size));
  r0 = GET0(z);
  r2 = GET2(z);
  get_odd_subview(z) = sin(get_odd_subview(a)) + cos(get_even_subview(a));
  r1 = sin(GET1(a)) + cos(GET0(a));
  r3 = sin(GET3(a)) + cos(GET2(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);
  //  even <-- even, odd
  Ramp<view_type>::apply(z, T(6), T(1)/T(size));
  r1 = GET1(z);
  r3 = GET3(z);
  get_even_subview(z) = sin(get_even_subview(a)) + cos(get_odd_subview(a));
  r0 = sin(GET0(a)) + cos(GET1(a));
  r2 = sin(GET2(a)) + cos(GET3(a));
  CHECK0(z, r);
  CHECK1(z, r);
  CHECK2(z, r);
  CHECK3(z, r);

  // binary(unary(subview1), unary(subview2)), aliasing two inputs and output
  //  odd <-- odd, even
  Ramp<view_type>::apply(a, T(5), T(1)/T(size));
  r0 = GET0(a);
  r2 = GET2(a);
  r1 = Get<view_type>::apply(a, i1) +
       Get<view_type>::apply(a, i1-1);
  r3 = Get<view_type>::apply(a, i3) +
       Get<view_type>::apply(a, i3-1);
  get_odd_subview(a) = get_odd_subview(a) + get_even_subview(a);
  CHECK0(a, r);
  CHECK1(a, r);
  CHECK2(a, r);
  CHECK3(a, r);
}



/// Non-elementwise tests for compound expressions compute results
/// both as single-line operations and then as multi-line expressions
/// involving temporaries.  Selected values are compared for accuracy.
///
template <typename T>
void
test_nonelementwise_vectors(length_type const N)
{
  OVXX_CT_ASSERT((is_same<typename scalar_of<T>::type, T>::value));
  typedef std::complex<T> C;
  typedef Vector<T> view_type;
  typedef Vector<C> cview_type;

  index_type i0 = 3;
  index_type i1 = N / 2 - 1;
  index_type i2 = N - 3;

#if DEBUG
  std::cout << __PRETTY_FUNCTION__
            << " size = " << N
            << std::endl;
#endif
  {
    cview_type a(N, C());
    cview_type b(N, C());
    cview_type y(N, C());
    cview_type z(N, C());
    cview_type x(N, C());
    view_type r(N, T());
    view_type s(N, T());
    C cr0 = C();
    C cr1 = C();
    C cr2 = C();

    // inputs
    a = ramp(C(1), C(1)/C(N), N);
    b = ramp(C(2), C(1)/C(N), N);
    r = ramp(T(1), T(1)/T(N), N);

    // outputs
    y = C();
    z = C();

    // temporary (input and output)
    s = T();
    x = C();

    typedef vsip::Fft<const_Vector, C, C, fft_fwd> fwd_fft_type;
    typedef vsip::Fft<const_Vector, C, C, fft_inv> inv_fft_type;

    fwd_fft_type f_fft(Domain<1>(N), 1.0);
    inv_fft_type i_fft(Domain<1>(N), 1.0 / N);
   
    // ne == non-elementwise operator

    // ne(unary(view))
    z = f_fft(sq(a));
    x = sq(a);
    y = f_fft(x);
    cr0 = y.get(i0);
    cr1 = y.get(i1);
    cr2 = y.get(i2);
    Check<cview_type, 1>::eval(z, i0, cr0, __LINE__);
    Check<cview_type, 1>::eval(z, i1, cr1, __LINE__);
    Check<cview_type, 1>::eval(z, i2, cr2, __LINE__);

    // unary(ne(unary(view)))
    z = mag(f_fft(sq(a)));
    x = sq(a);
    y = f_fft(x);
    y = mag(y);
    cr0 = y.get(i0);
    cr1 = y.get(i1);
    cr2 = y.get(i2);
    Check<cview_type, 1>::eval(z, i0, cr0, __LINE__);
    Check<cview_type, 1>::eval(z, i1, cr1, __LINE__);
    Check<cview_type, 1>::eval(z, i2, cr2, __LINE__);

    // ne(binary(ne(unary(view)), view))
    z = i_fft(f_fft(sq(a)) * b);
    y = sq(a);
    y = f_fft(y);
    y = y * b;
    y = i_fft(y);
    cr0 = y.get(i0);
    cr1 = y.get(i1);
    cr2 = y.get(i2);
    Check<cview_type, 1>::eval(z, i0, cr0, __LINE__);
    Check<cview_type, 1>::eval(z, i1, cr1, __LINE__);
    Check<cview_type, 1>::eval(z, i2, cr2, __LINE__);

    // ne(binary(view, ne(unary(view))))
    z = i_fft(b * f_fft(sq(a)));
    y = sq(a);
    x = f_fft(y);
    x *= b;
    y = i_fft(x);
    cr0 = y.get(i0);
    cr1 = y.get(i1);
    cr2 = y.get(i2);
    Check<cview_type, 1>::eval(z, i0, cr0, __LINE__);
    Check<cview_type, 1>::eval(z, i1, cr1, __LINE__);
    Check<cview_type, 1>::eval(z, i2, cr2, __LINE__);

    // ne(binary(ne(unary(view)), view)), real filter values
    z = i_fft(f_fft(sq(a)) * r);
    x = sq(a);
    x = f_fft(x);
    x *= r;
    y = i_fft(x);
    cr0 = y.get(i0);
    cr1 = y.get(i1);
    cr2 = y.get(i2);
    Check<cview_type, 1>::eval(z, i0, cr0, __LINE__);
    Check<cview_type, 1>::eval(z, i1, cr1, __LINE__);
    Check<cview_type, 1>::eval(z, i2, cr2, __LINE__);
  }


  // Complex/Real mixed FFT cases
  {
    length_type const N2 = (N/2)+1;
    view_type a(N, T());
    view_type b(N, T());
    view_type y(N, T());
    view_type z(N, T());
    cview_type yy(N2, C());
    cview_type zz(N2, C());
    view_type x(N, T());
    view_type r(N, T());
    view_type s(N, T());
    T r0 = T();
    T r1 = T();
    T r2 = T();
    C cr0 = C();
    C cr1 = C();
    C cr2 = C();

    // inputs
    a = ramp(T(1), T(1)/T(N), N);
    b = ramp(T(2), T(1)/T(N), N);
    r = ramp(T(1), T(1)/T(N), N);

    // outputs
    y = T();
    z = T();
    yy = C();
    zz = C();

    // temporary (input and output)
    s = T();
    x = T();

    typedef vsip::Fft<const_Vector, T, C> fwd_fft_type;
    typedef vsip::Fft<const_Vector, C, T> inv_fft_type;

    fwd_fft_type f_fft(Domain<1>(N), 1.0);
    inv_fft_type i_fft(Domain<1>(N), 1.0 / N);
   
    // ne == non-elementwise operator

    // ne(unary(view))
    zz = f_fft(sq(a));
    x = sq(a);
    yy = f_fft(x);
    cr0 = yy.get(i0/2);  // index is reduced to stay in valid range
    cr1 = yy.get(i1/2);  //   due to there only being N/2-1 samples.
    cr2 = yy.get(i2/2);
    Check<cview_type, 1>::eval(zz, i0/2, cr0, __LINE__);
    Check<cview_type, 1>::eval(zz, i1/2, cr1, __LINE__);
    Check<cview_type, 1>::eval(zz, i2/2, cr2, __LINE__);

    // ne(binary(ne(unary(view)), view))
    z = i_fft(f_fft(sq(a)) * b);
    x = sq(a);
    yy = f_fft(x);
    yy *= b;
    y = i_fft(yy);
    r0 = y.get(i0);
    r1 = y.get(i1);
    r2 = y.get(i2);
    Check<view_type, 1>::eval(z, i0, r0, __LINE__);
    Check<view_type, 1>::eval(z, i1, r1, __LINE__);
    Check<view_type, 1>::eval(z, i2, r2, __LINE__);
  }
}


template <typename T>
void
test_elementwise_cases(
  length_type const M,
  length_type const N,
  length_type const P)
{
  test_elementwise<1, T>(Domain<1>(M * N * P));
  test_elementwise<2, T>(Domain<2>(M * N, P));
  test_elementwise<3, T>(Domain<3>(M, N, P));

  test_elementwise_subview<1, T>(Domain<1>(M * N * P));
  test_elementwise_subview<2, T>(Domain<2>(M * N, P));
  test_elementwise_subview<3, T>(Domain<3>(M, N, P));
}


