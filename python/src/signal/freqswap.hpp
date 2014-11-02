//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef signal_freqswap_hpp_
#define signal_freqswap_hpp_

#include <ovxx/python/block.hpp>
#include <ovxx/view.hpp>
#include <vsip/signal.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <vsip::dimension_type D, typename T>
void freqswap(ovxx::python::Block<D, T> const &in, ovxx::python::Block<D, T> &out)
{
  typedef ovxx::python::Block<D, T> B;
  typedef typename ovxx::view_of<B>::type V;
  ovxx::signal::freqswap(V(const_cast<B&>(in)), V(out));
}

}

#endif
