//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_reductions_reductions_idx_hpp_
#define ovxx_reductions_reductions_idx_hpp_

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <ovxx/reductions/functors.hpp>
#include <ovxx/dispatch.hpp>
#if OVXX_HAVE_CVSIP
# include <ovxx/cvsip/reductions_idx.hpp>
#endif

namespace ovxx
{
namespace dispatcher
{

template<template <typename> class R>
struct List<op::reduce_idx<R> >
{
  typedef make_type_list<be::parallel, be::cvsip, be::cuda, be::generic>::type type;
};

/// Generic evaluator for vector reductions.
template <template <typename> class R,
          typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<1>&, row1_type)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<1>&, row1_type)
  { return true; }

  static void exec(T& r, Block const& a, Index<1>& idx, row1_type)
  {
    index_type maxi = 0;

    R<typename Block::value_type> maxv(a.get(maxi));

    for (index_type i=0; i<a.size(1, 0); ++i)
      if (maxv.next_value(a.get(i)))
	maxi = i;

    idx = Index<1>(maxi);
    r =  maxv.value();
  }
};


/// Generic evaluator for matrix reductions (tuple<0, 1, 2>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<2>&, row2_type)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<2>&, row2_type)
  { return true;}

  static void exec(T& r, Block const& a, Index<2>& idx, row2_type)
  {
    index_type maxi = 0;
    index_type maxj = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj));

    for (index_type i=0; i<a.size(2, 0); ++i)
    for (index_type j=0; j<a.size(2, 1); ++j)
    {
      if (maxv.next_value(a.get(i, j)))
      {
	maxi = i;
	maxj = j;
      }
    }
  
    idx = Index<2>(maxi, maxj);
    r   = maxv.value();
  }
};


/// Generic evaluator for matrix reductions (tuple<2, 1, 0>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<2>&, col2_type)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<2>&, col2_type)
  { return true; }

  static void exec(T& r, Block const& a, Index<2>& idx, col2_type)
  {
    index_type maxi = 0;
    index_type maxj = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj));

    for (index_type j=0; j<a.size(2, 1); ++j)
    for (index_type i=0; i<a.size(2, 0); ++i)
    {
      if (maxv.next_value(a.get(i, j)))
      {
	maxi = i;
	maxj = j;
      }
    }
  
    idx = Index<2>(maxi, maxj);
    r   = maxv.value();
  }
};


/// Generic evaluator for tensor reductions (tuple<0, 1, 2>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<3>&, tuple<0, 1, 2>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<3>&, tuple<0, 1, 2>)
  { return true;}

  static void exec(T& r, Block const& a, Index<3>& idx, tuple<0, 1, 2>)
  { 
    index_type maxi = 0;
    index_type maxj = 0;
    index_type maxk = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj, maxk));

    for (index_type i=0; i<a.size(3, 0); ++i)
    for (index_type j=0; j<a.size(3, 1); ++j)
    for (index_type k=0; k<a.size(3, 2); ++k)
    {
      if (maxv.next_value(a.get(i, j, k)))
      {
	maxi = i;
	maxj = j;
	maxk = k;
      }
    }

    idx = Index<3>(maxi, maxj, maxk);
    r   = maxv.value();
  }
};


/// Generic evaluator for tensor reductions (tuple<0, 2, 1>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<3>&, tuple<0, 2, 1>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<3>&, tuple<0, 2, 1>)
  { return true; }

  static void exec(T& r, Block const& a, Index<3>& idx, tuple<0, 2, 1>)
  {
    index_type maxi = 0;
    index_type maxj = 0;
    index_type maxk = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj, maxk));

    for (index_type i=0; i<a.size(3, 0); ++i)
    for (index_type k=0; k<a.size(3, 2); ++k)
    for (index_type j=0; j<a.size(3, 1); ++j)
    {
      if (maxv.next_value(a.get(i, j, k)))
      {
	maxi = i;
	maxj = j;
	maxk = k;
      }
    }

    idx = Index<3>(maxi, maxj, maxk);
    r   = maxv.value();
  }
};


/// Generic evaluator for tensor reductions (tuple<1, 0, 2>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<3>&, tuple<1, 0, 2>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<3>&, tuple<1, 0, 2>)
  { return true;}

  static void exec(T& r, Block const& a, Index<3>& idx, tuple<1, 0, 2>)
  {
    index_type maxi = 0;
    index_type maxj = 0;
    index_type maxk = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj, maxk));

    for (index_type j=0; j<a.size(3, 1); ++j)
    for (index_type i=0; i<a.size(3, 0); ++i)
    for (index_type k=0; k<a.size(3, 2); ++k)
    {
      if (maxv.next_value(a.get(i, j, k)))
      {
	maxi = i;
	maxj = j;
	maxk = k;
      }
    }

    idx = Index<3>(maxi, maxj, maxk);
    r   = maxv.value();
  }
};


/// Generic evaluator for tensor reductions (tuple<1, 2, 0>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<3>&, tuple<1, 2, 0>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<3>&, tuple<1, 2, 0>)
  { return true;}

  static void exec(T& r, Block const& a, Index<3>& idx, tuple<1, 2, 0>)
  {
    index_type maxi = 0;
    index_type maxj = 0;
    index_type maxk = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj, maxk));

    for (index_type j=0; j<a.size(3, 1); ++j)
    for (index_type k=0; k<a.size(3, 2); ++k)
    for (index_type i=0; i<a.size(3, 0); ++i)
    {
      if (maxv.next_value(a.get(i, j, k)))
      {
	maxi = i;
	maxj = j;
	maxk = k;
      }
    }

    idx = Index<3>(maxi, maxj, maxk);
    r   = maxv.value();
  }
};


/// Generic evaluator for tensor reductions (tuple<2, 0, 1>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<3>&, tuple<2, 0, 1>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<3>&, tuple<2, 0, 1>)
  { return true; }

  static void exec(T& r, Block const& a, Index<3>& idx, tuple<2, 0, 1>)
  {
    index_type maxi = 0;
    index_type maxj = 0;
    index_type maxk = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj, maxk));

    for (index_type k=0; k<a.size(3, 2); ++k)
    for (index_type i=0; i<a.size(3, 0); ++i)
    for (index_type j=0; j<a.size(3, 1); ++j)
    {
      if (maxv.next_value(a.get(i, j, k)))
      {
	maxi = i;
	maxj = j;
	maxk = k;
      }
    }

    idx = Index<3>(maxi, maxj, maxk);
    r   = maxv.value();
  }
};


/// Generic evaluator for tensor reductions (tuple<2, 1, 0>).
template <template <typename> class R,
	  typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::generic,
                 void(T&, Block const&, Index<3>&, tuple<2, 1, 0>)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T&, Block const&, Index<3>&, tuple<2, 1, 0>)
  { return true; }

  static void exec(T& r, Block const& a, Index<3>& idx, tuple<2, 1, 0>)
  {
    index_type maxi = 0;
    index_type maxj = 0;
    index_type maxk = 0;

    R<typename Block::value_type> maxv(a.get(maxi, maxj, maxk));

    for (index_type k=0; k<a.size(3, 2); ++k)
    for (index_type j=0; j<a.size(3, 1); ++j)
    for (index_type i=0; i<a.size(3, 0); ++i)
    {
      if (maxv.next_value(a.get(i, j, k)))
      {
	maxi = i;
	maxj = j;
	maxk = k;
      }
    }

    idx = Index<3>(maxi, maxj, maxk);
    r   = maxv.value();
  }
};

} // namespace ovxx::dispatcher

template <template <typename> class R,
	  typename V>
typename R<typename V::value_type>::result_type
reduce_idx(V view, Index<V::dim> &idx)
{
  using namespace dispatcher;

  typedef typename V::value_type T;
  typedef typename R<T>::result_type result_type;
  typedef typename V::block_type block_type;
  typedef Index<V::dim> index_type;
  typedef typename get_block_layout<typename V::block_type>::order_type
    order_type;

  result_type r;

  dispatch<op::reduce_idx<R>, void,
    result_type&, block_type const&, index_type&, order_type>
    (r, view.block(), idx, order_type());

  return r;
}

} // namespace ovxx
	     
#endif
