//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dispatch_hpp_
#define ovxx_dispatch_hpp_

#include <ovxx/dispatcher/tags.hpp>
#include <ovxx/dispatcher/dispatcher.hpp>
#include <ovxx/dispatcher/diagnostics.hpp>

namespace ovxx
{

// If we can use variadic templates, life is easy...
#ifdef __GXX_EXPERIMENTAL_CXX0X__

template <typename O, typename R, typename ...A>
R dispatch(A... args) 
{
  typedef dispatcher::Dispatcher<O, R(A...)> dispatcher_type;
  return dispatcher_type::dispatch(args...);
}

#else

template <typename O, typename R, typename A>
R dispatch(A a) 
{
  typedef dispatcher::Dispatcher<O, R(A)> dispatcher_type;
  return dispatcher_type::dispatch(a);
}

template <typename O, typename R, typename A1, typename A2>
R dispatch(A1 a1, A2 a2) 
{ 
  typedef dispatcher::Dispatcher<O, R(A1, A2)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2);
}

template <typename O, typename R, typename A1, typename A2, typename A3>
R dispatch(A1 a1, A2 a2, A3 a3) 
{
  typedef dispatcher::Dispatcher<O, R(A1, A2, A3)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2, a3);
}

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4>
R dispatch(A1 a1, A2 a2, A3 a3, A4 a4) 
{
  typedef dispatcher::Dispatcher<O, R(A1, A2, A3, A4)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2, a3, a4);
}

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5>
R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) 
{
  typedef dispatcher::Dispatcher<O, R(A1, A2, A3, A4, A5)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2, a3, a4, a5);
}

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6>
R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) 
{
  typedef dispatcher::Dispatcher<O, R(A1, A2, A3, A4, A5, A6)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2, a3, a4, a5, a6);
}

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6, typename A7>
R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7) 
{
  typedef dispatcher::Dispatcher<O, R(A1, A2, A3, A4, A5, A6, A7)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2, a3, a4, a5, a6, a7);
}

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6, typename A7, typename A8>
R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8) 
{
  typedef dispatcher::Dispatcher<O, R(A1, A2, A3, A4, A5, A6, A7, A8)> dispatcher_type;
  return dispatcher_type::dispatch(a1, a2, a3, a4, a5, a6, a7, a8);
}

#endif

} // namespace ovxx

#endif
