//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef elementwise_hpp_
#define elementwise_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

#define REDUCE(F)					\
template <dimension_type D, typename T>		        \
inline T F(Block<D, T> const &b)			\
{							\
  typedef Block<D, T> B;				\
  typename view_of<B>::const_type v(const_cast<B&>(b)); \
  Domain<D> dom = ovxx::block_domain<D>(b);		\
  return vsip::F(v);					\
}

#define REDUCE_I(F)					\
template <dimension_type D, typename T>		        \
inline bpl::object F(Block<D, T> const &b)		\
{							\
  typedef Block<D, T> B;				\
  typename view_of<B>::const_type v(const_cast<B&>(b)); \
  vsip::Index<D> i;					\
  Domain<D> dom = ovxx::block_domain<D>(b);		\
  T r = vsip::F(v, i);					\
  return bpl::make_tuple(r, i);				\
}

REDUCE(meanval)
REDUCE_I(maxval)
REDUCE_I(minval)
REDUCE(sumval)

template <dimension_type D, typename T>
void define_reductions()
{
  typedef Block<D, T> block_type;
  bpl::def("meanval", meanval<D, T>);
  bpl::def("maxval", maxval<D, T>);
  bpl::def("minval", minval<D, T>);
  bpl::def("sumval", sumval<D, T>);
}

} // namespace pyvsip

#endif
