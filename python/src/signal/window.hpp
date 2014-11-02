//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef signal_window_hpp_
#define signal_window_hpp_

#include <ovxx/python/block.hpp>
#include <ovxx/signal/window.hpp>
#include <ovxx/domain_utils.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <typename T>
bpl::object blackman(vsip::length_type len)
{
  typedef ovxx::python::Block<1, T> B;
  vsip::Vector<T> v = ovxx::signal::blackman<T>(len);
  boost::shared_ptr<B> result(new B(ovxx::block_domain<1>(v.block())));
  ovxx::assign<1>(*result, v.block());
  return bpl::object(result);
}

template <typename T>
bpl::object cheby(vsip::length_type len, float ripple)
{
  typedef ovxx::python::Block<1, T> B;
  vsip::Vector<T> v = ovxx::signal::cheby<T>(len, ripple);
  B *result = new B(ovxx::block_domain<1>(v.block()));
  ovxx::assign<1>(*result, v.block());
  return bpl::object(boost::shared_ptr<B>(result));
}

template <typename T>
bpl::object hanning(vsip::length_type len)
{
  typedef ovxx::python::Block<1, T> B;
  vsip::Vector<T> v = ovxx::signal::hanning<T>(len);
  B *result = new B(ovxx::block_domain<1>(v.block()));
  ovxx::assign<1>(*result, v.block());
  return bpl::object(boost::shared_ptr<B>(result));
}

template <typename T>
bpl::object kaiser(vsip::length_type len, float beta)
{
  typedef ovxx::python::Block<1, T> B;
  vsip::Vector<T> v = ovxx::signal::kaiser<T>(len, beta);
  B *result = new B(ovxx::block_domain<1>(v.block()));
  ovxx::assign<1>(*result, v.block());
  return bpl::object(boost::shared_ptr<B>(result));
}
  
}




#endif
