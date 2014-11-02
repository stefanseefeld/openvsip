//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef selgen_clip_hpp_
#define selgen_clip_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/selgen.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <vsip::dimension_type D, typename T>
void clip(ovxx::python::Block<D, T> const &bin, ovxx::python::Block<D, T> &bout, T lt, T ut, T lc, T uc)
{
  typedef ovxx::python::Block<D, T> B;
  typedef typename ovxx::view_of<B>::type V;
  V vout(bout);
  vout = vsip::clip(V(const_cast<B&>(bin)), lt, ut, lc, uc);
}

template <vsip::dimension_type D, typename T>
void invclip(ovxx::python::Block<D, T> const &bin, ovxx::python::Block<D, T> &bout, T lt, T mt, T ut, T lc, T uc)
{
  typedef ovxx::python::Block<D, T> B;
  typedef typename ovxx::view_of<B>::type V;
  V vout(bout);
  vout = vsip::invclip(V(const_cast<B&>(bin)), lt, mt, ut, lc, uc);
}

template <typename T>
void define()
{
  bpl::def("clip", clip<1, T>);
  bpl::def("clip", clip<2, T>);
  bpl::def("invclip", invclip<1, T>);
  bpl::def("invclip", invclip<2, T>);
}


}

#endif
