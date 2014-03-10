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

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

template <dimension_type D, typename T>
bpl::object eq(Block<D, T> const &b1, Block<D, T> const &b2)
{
  typedef Block<D, T> B;
  typename view_of<B>::const_type v1(const_cast<B&>(b1));
  typename view_of<B>::const_type v2(const_cast<B&>(b2));
  Domain<D> dom = ovxx::block_domain<D>(b1);
  Block<D, bool> *result = new Block<D, bool>(dom);
  typename view_of<Block<D, bool> >::type r(*result);
  r = vsip::eq(v1, v2);
  return bpl::object(boost::shared_ptr<Block<D, bool> >(result));
}


template <dimension_type D, typename T>
void define_elementwise()
{
  typedef Block<D, T> block_type;
  bpl::def("eq", eq<D, T>);
}

} // namespace pyvsip

#endif
