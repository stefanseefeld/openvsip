/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/selgen/selection.cpp
    @author  Stefan Seefeld
    @date    2009-09-09
    @brief   VSIPL++ Library: Python bindings for selection types and functions.

*/
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/selgen.hpp>
#include <vsip/vector.hpp>
#include "../block.hpp"

namespace pyvsip
{
template <typename T1, typename T2>
vsip::index_type
first(vsip::index_type begin, bpl::object predicate,
      vsip::Vector<T1, Block<1, T1> > v,
      vsip::Vector<T2, Block<1, T2> > w)
{
  if (begin > v.size()) 
    return begin;
  for (vsip::index_type i = begin, end = v.size(); i != end; ++i)
    if (predicate(v.get(i), w.get(i)))
      return i;
  return v.size();
}

template <typename T>
bpl::tuple indexbool1(vsip::Vector<T, Block<1, T> > source)
{
  vsip::Vector<vsip::Index<1>, Block<1, vsip::Index<1> > > 
    indices(source.length());
  vsip::length_type index = vsip::indexbool(source, indices);
  return bpl::make_tuple(index, indices);
}

template <typename T>
bpl::tuple indexbool2(vsip::Matrix<T, Block<2, T> > source)
{
  vsip::Vector<vsip::Index<2>, Block<1, vsip::Index<2> > >
    indices(source.size());
  vsip::length_type index = vsip::indexbool(source, indices);
  return bpl::make_tuple(index, indices);
}

template <typename T>
vsip::Vector<T, Block<1, T> >
gather1(vsip::Vector<T, Block<1, T> > source,
	vsip::Vector<vsip::Index<1>, Block<1, vsip::Index<1> > > indices)
{
  vsip::Vector<T, Block<1, T> > result(indices.size());
  for (vsip::index_type i = 0; i != indices.size(); ++i)
    result.put(i, get(source, indices.get(i)));
  return result;
}

template <typename T>
vsip::Vector<T, Block<1, T> >
gather2(vsip::Matrix<T, Block<1, T> > source,
	vsip::Vector<vsip::Index<2>, Block<1, vsip::Index<2> > > indices)
{
  vsip::Vector<T, Block<1, T> > result(indices.size());
  for (vsip::index_type i = 0; i != indices.size(); ++i)
    result.put(i, get(source, indices.get(i)));
  return result;
}

template <typename T>
void
scatter1(vsip::Vector<T, Block<1, T> > source,
	 vsip::Vector<vsip::Index<1>, Block<1, vsip::Index<1> > > indices,
	 vsip::Vector<T, Block<1, T> > destination)
{ scatter(source, indices, destination);}

template <typename T>
void
scatter2(vsip::Vector<T, Block<1, T> > source,
	 vsip::Vector<vsip::Index<2>, Block<1, vsip::Index<2> > > indices,
	 vsip::Matrix<T, Block<2, T> > destination)
{ scatter(source, indices, destination);}


template <typename T>
void define_selection()
{
  bpl::def("indexbool", indexbool1<T>);
  bpl::def("indexbool", indexbool2<T>);
  bpl::def("gather", gather1<T>);
  bpl::def("gather", gather2<T>);
  bpl::def("scatter", scatter1<T>);
  bpl::def("scatter", scatter2<T>);
}

}

BOOST_PYTHON_MODULE(selection)
{
  using namespace pyvsip;

  //bpl::def("first", first<float, float>);
  bpl::def("first", first<double, double>);

  //define_selection<float>();
  define_selection<double>();
}
