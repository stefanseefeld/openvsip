/* Copyright (c) 2010, by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Common routines for math reduction tests.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/math.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>


/***********************************************************************
  sumval tests.
***********************************************************************/

template <typename T> struct widerT { typedef T type; };

template <> struct widerT<char>           { typedef short             type; };
template <> struct widerT<signed char>    { typedef signed short      type; };
template <> struct widerT<unsigned char>  { typedef unsigned short    type; };
template <> struct widerT<short>          { typedef int               type; };
template <> struct widerT<unsigned short> { typedef unsigned int      type; };
template <> struct widerT<int>            { typedef long int          type; };
template <> struct widerT<unsigned int>   { typedef unsigned long int type; };

template <typename ViewT>
void
view_sumval(
  ViewT       view,
  vsip::length_type count)
{
  using namespace vsip;
  using namespace vsip_csl;
  typedef typename ViewT::value_type T;

  typedef typename widerT<T>::type W;

  view = T();

  index_type  i        = 0;
  T           expected = T();
  W          wexpected = W();
  length_type size     = view.size();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = T(i) - T(5);

    expected -= get_nth(view, i);
    expected += nval;

    wexpected -= get_nth(view, i);
    wexpected += nval;

    put_nth(view, i, nval);
    
    T val = sumval(view);
    test_assert(vsip_csl::equal(val, expected));
    
    W wval = sumval(view, W());
    test_assert(vsip_csl::equal(wval, wexpected));
  }
}



template <typename             StoreT,
          vsip::dimension_type Dim>
void
test_sumval(vsip::Domain<Dim> const& dom, vsip::length_type count)
{
  typedef typename StoreT::value_type T;

  StoreT      store(dom, T());

  view_sumval(store.view, count);
}



template <typename T>
void
cover_sumval()
{
  using namespace vsip;
  test_sumval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_sumval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_sumval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_sumval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_sumval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_sumval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_sumval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_sumval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_sumval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);

  test_sumval<Storage<1, T, row1_type, Map<Block_dist> > >(Domain<1>(15), 8);
  test_sumval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
}



template <typename T,
	  typename MapT>
void
par_cover_sumval()
{
  using namespace vsip;
  typedef Dense<1, T, row1_type, MapT> block_type;
  typedef Vector<T, block_type>        view_type;

  length_type size = 8;

  MapT      map = create_map<1, MapT>();
  view_type view(size, map);

  view_sumval(view, 8);
}



/***********************************************************************
  sumval bool tests.
***********************************************************************/

template <typename             StoreT,
	  vsip::dimension_type Dim>
void
test_sumval_bool(vsip::Domain<Dim> const& dom, vsip::length_type count)
{
  using namespace vsip;
  using namespace vsip_csl;
  StoreT      store(dom, false);
  length_type size = store.view.size();

  index_type  i        = 0;
  length_type expected = 0;
  
  for (index_type c=0; c<count; ++c)
  {
    i         = (2*i+3) % size;
    bool nval = (3*i+1) % 2 == 0;

    if (get_nth(store.view, i))
      expected -= 1;

    if (nval)
      expected += 1;

    put_nth(store.view, i, nval);
    
    length_type val = sumval(store.view);
    test_assert(vsip_csl::equal(val, expected));
  }
}



/***********************************************************************
  sumsqval tests.

  Note: sumsqval returns the sum of squares of elements of a view.
***********************************************************************/

template <typename             StoreT,
          vsip::dimension_type Dim>
void
test_sumsqval(vsip::Domain<Dim> const& dom, vsip::length_type count)
{
  using namespace vsip;
  using namespace vsip_csl;
  typedef typename StoreT::value_type T;

  typedef typename widerT<T>::type W;

  StoreT      store(dom, T());
  length_type size = store.view.size();

  index_type  i        = 0;
  T           expected = T();
  W          wexpected = W();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = T(i) - T(5);

    T nth  = get_nth(store.view, i);
    expected -= (nth  * nth);
    expected += (nval * nval);
    wexpected -= (nth  * nth);
    wexpected += (nval * nval);

    put_nth(store.view, i, nval);
    
    T val = sumsqval(store.view);
    test_assert(equal(val, expected));
    
    W wval = sumsqval(store.view, W());
    test_assert(vsip_csl::equal(wval, wexpected));
  }
}



template <typename T>
void
cover_sumsqval()
{
  using namespace vsip;
  test_sumsqval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_sumsqval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_sumsqval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_sumsqval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_sumsqval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_sumsqval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_sumsqval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_sumsqval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_sumsqval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);

  test_sumsqval<Storage<1, T, row1_type, Map<Block_dist> > >(Domain<1>(15), 8);
  test_sumsqval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
}



/***********************************************************************
  meanval tests.
***********************************************************************/

template <typename             StoreT,
          vsip::dimension_type Dim>
void
test_meanval(vsip::Domain<Dim> const& dom, vsip::length_type count)
{
  using namespace vsip;
  using namespace vsip_csl;
  typedef typename StoreT::value_type T;
  typedef typename widerT<T>::type W;

  StoreT      store(dom, T());
  length_type size = store.view.size();

  index_type  i        = 0;
  T           expected = T();
  W          wexpected = W();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = T(i) - T(5);

    expected -= get_nth(store.view, i);
    expected += nval;
    wexpected -= get_nth(store.view, i);
    wexpected += nval;

    put_nth(store.view, i, nval);
    
    T sval = sumval(store.view);
    T mval = meanval(store.view);
    test_assert(vsip_csl::equal(sval, expected));
    test_assert(vsip_csl::equal(mval, T(expected/static_cast<T>(store.view.size()))));
    
    W wsval = sumval(store.view, W());
    W wmval = meanval(store.view, W());
    test_assert(vsip_csl::equal(wsval, wexpected));
    test_assert(vsip_csl::equal(wmval, W(wexpected/static_cast<W>(store.view.size()))));
  }
}



template <typename T>
void
cover_meanval()
{
  using namespace vsip;
  test_meanval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_meanval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_meanval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_meanval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_meanval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_meanval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_meanval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_meanval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_meanval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);

  test_meanval<Storage<1, T, row1_type, Map<Block_dist> > >(Domain<1>(15), 8);
  test_meanval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
}



/***********************************************************************
  meansqval tests.
***********************************************************************/

template <typename           StoreT,
          vsip::dimension_type Dim>
void
test_meansqval(vsip::Domain<Dim> const& dom, vsip::length_type count)
{
  using namespace vsip;
  using namespace vsip_csl;
  typedef typename StoreT::value_type T;
  typedef typename vsip::impl::scalar_of<T>::type R;
  typedef typename widerT<R>::type W;

  StoreT      store(dom, T());
  length_type size = store.view.size();

  index_type  i        = 0;
  W wexpected = W();
  
  for (index_type c=0; c<count; ++c)
  {
    i      = (2*i+3) % size;
    T nval = T(i) - T(5);

    T nth  = get_nth(store.view, i);

    wexpected -= vsip::impl::fn::magsq(nth, W());
    wexpected += vsip::impl::fn::magsq(nval, W());

    put_nth(store.view, i, nval);
    
    W wmval = meansqval(store.view, W());
    test_assert(vsip_csl::equal(wmval, W(wexpected/static_cast<W>(store.view.size()))));
  }
}



template <typename T>
void
cover_meansqval()
{
  using namespace vsip;
  test_meansqval<Storage<1, T> >(Domain<1>(15), 8);
  
  test_meansqval<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_meansqval<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_meansqval<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_meansqval<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_meansqval<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_meansqval<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_meansqval<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_meansqval<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);

  test_meansqval<Storage<1, T, row1_type, Map<Block_dist> > >(Domain<1>(15), 8);
  test_meansqval<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
}

