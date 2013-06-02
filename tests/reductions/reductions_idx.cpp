//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Tests for math reductions returning an index.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/map.hpp>
#include <test.hpp>
#include <storage.hpp>

using namespace ovxx;

template <typename T>
void
test_maxval_v(Domain<1> const& dom, length_type count)
{
  length_type size  = dom.size();;

  Vector<T>   vec(size, T());
  Index<1>    idx;

  index_type  i = 0;
  T           val = T();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = val + T(1);
    vec(i) = nval;
    
    val = maxval(vec, idx);
    test_assert(equal(val, nval));
    test_assert(idx == i);
  }
}



template <typename       StoreT,
	  dimension_type Dim>
void
test_maxval(Domain<Dim> const& dom, length_type count)
{
  typedef typename StoreT::value_type T;

  StoreT      store(dom, T());
  Index<Dim>  idx;
  length_type size = store.view.size();

  index_type  i   = 0;
  T           val = T();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = val + T(1);

    put_nth(store.view, i, nval);
    
    val = maxval(store.view, idx);
    test_assert(equal(val, nval));
    test_assert(nth_from_index(store.view, idx) == i);
  }
}



template <typename T>
void
cover_maxval()
{
  test_maxval_v<T>(Domain<1>(15), 8);

  test_maxval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_maxval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_maxval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_maxval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_maxval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_maxval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_maxval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_maxval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_maxval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);
#if OVXX_PARALLEL
  test_maxval<Storage<1, T, row1_type, Map<> > >(Domain<1>(15), 8);
  test_maxval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
#endif
}



/***********************************************************************
  minval tests.
***********************************************************************/

template <typename       StoreT,
	  dimension_type Dim>
void
test_minval(Domain<Dim> const& dom, length_type count)
{
  typedef typename StoreT::value_type T;

  StoreT      store(dom, T());
  Index<Dim>  idx;
  length_type size = store.view.size();

  index_type  i   = 0;
  T           val = T();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = val - T(1);

    put_nth(store.view, i, nval);
    
    val = minval(store.view, idx);
    test_assert(equal(val, nval));
    test_assert(nth_from_index(store.view, idx) == i);
  }
}



template <typename T>
void
cover_minval()
{
  test_minval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_minval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_minval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_minval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_minval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_minval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_minval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_minval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_minval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);
#if OVXX_PARALLEL
  test_minval<Storage<1, T, row1_type, Map<> > >(Domain<1>(15), 8);
  test_minval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
#endif
}



/***********************************************************************
  {min,max}mgval tests.
***********************************************************************/

template <typename       StoreT,
	  dimension_type Dim>
void
test_mgval(Domain<Dim> const& dom, length_type count)
{
  typedef typename StoreT::value_type T;
  typedef typename scalar_of<T>::type scalar_type;

  StoreT      store(dom, T(30, 40));
  Index<Dim>  idx;
  length_type size = store.view.size();

  index_type  i     = 0;
  T           large = T(30, 40);
  T           small = T(30, 40);
  
  for (index_type c=0; c<count; ++c)
  {
    i            = (2*i+3) % size;
    index_type j = (i+1)   % size;

    large        = large + T(3, 4);
    small        = small - T(0.3, 0.4);

    put_nth(store.view, i, large);
    put_nth(store.view, j, small);
    
    scalar_type lval = maxmgval(store.view, idx);
    test_assert(equal(lval, mag(large)));
    test_assert(nth_from_index(store.view, idx) == i);

    scalar_type sval = minmgval(store.view, idx);
    test_assert(equal(sval, mag(small)));
    test_assert(nth_from_index(store.view, idx) == j);
  }
}



template <typename T>
void
cover_mgval()
{
  test_mgval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_mgval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_mgval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_mgval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_mgval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_mgval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_mgval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_mgval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_mgval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);
#if OVXX_PARALLEL
  test_mgval<Storage<1, T, row1_type, Map<> > >(Domain<1>(15), 8);
  test_mgval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
#endif
}



/***********************************************************************
  {min,max}mgsqval tests.
***********************************************************************/

template <typename       StoreT,
	  dimension_type Dim>
void
test_mgsqval(Domain<Dim> const& dom, length_type count)
{
  typedef typename StoreT::value_type T;
  typedef typename scalar_of<T>::type scalar_type;

  StoreT      store(dom, T(3, 4));
  Index<Dim>  idx;
  length_type size = store.view.size();

  index_type  i     = 0;
  T           large = T(3, 4);
  T           small = T(3, 4);
  
  for (index_type c=0; c<count; ++c)
  {
    i            = (2*i+3) % size;
    index_type j = (i+1)   % size;

    large        = large + T(3, 4);
    small        = small - T(0.01, 0.01);

    put_nth(store.view, i, large);
    put_nth(store.view, j, small);
    
    scalar_type lval = maxmgsqval(store.view, idx);
    test_assert(equal(lval, magsq(large)));
    test_assert(nth_from_index(store.view, idx) == i);

    scalar_type sval = minmgsqval(store.view, idx);
    test_assert(equal(sval, magsq(small)));
    test_assert(nth_from_index(store.view, idx) == j);
  }
}



template <typename T>
void
cover_mgsqval()
{
  test_mgsqval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_mgsqval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_mgsqval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_mgsqval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_mgsqval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_mgsqval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_mgsqval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_mgsqval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_mgsqval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);
#if OVXX_PARALLEL
  test_mgsqval<Storage<1, T, row1_type, Map<> > >(Domain<1>(15), 8);
  test_mgsqval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
#endif
}





template <typename T>
void
simple_mgval_c()
{
   length_type         size = 13;
   Vector<complex<T> > vec(size, complex<T>(3, 4));
   Index<1>            idx;
   T                   val;

   val = maxmgval(vec, idx);
   test_assert(equal(val, T(5)));
   // test_assert(idx == 0);

   val = minmgval(vec, idx);
   test_assert(equal(val, T(5)));
   // test_assert(idx == 0);


   vec(1) = complex<T>(6, 8);
   vec(2) = complex<T>(0.3, 0.4);

   val = maxmgval(vec, idx);
   test_assert(equal(val, T(10)));
   test_assert(idx == 1);

   val = minmgval(vec, idx);
   test_assert(equal(val, T(0.5)));
   test_assert(idx == 2);
}



int
main(int argc, char** argv)
{
   vsipl init(argc, argv);

   cover_maxval<int>();
   cover_maxval<float>();

   cover_minval<int>();
   cover_minval<float>();

   // cover_maxmgval<complex<int> >();
   cover_mgval<complex<float> >();

   // cover_maxmgsqval<complex<int> >();
   cover_mgsqval<complex<float> >();

   simple_mgval_c<float>();

#if VSIP_IMPL_TEST_DOUBLE
   cover_maxval<double>();
   cover_minval<double>();
   cover_mgval<complex<double> >();
   cover_mgsqval<complex<double> >();
#endif
}
