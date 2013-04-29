/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/dispatch.hpp
    @author  Stefan Seefeld
    @date    2006-11-03
    @brief   VSIPL++ Library: Dispatcher harness.
*/

#ifndef VSIP_OPT_DISPATCH_HPP
#define VSIP_OPT_DISPATCH_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/core/type_list.hpp>
#include <vsip/core/dispatch.hpp>
#include <vsip/opt/dispatch_profile.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip_csl
{
namespace dispatcher
{
using vsip::impl::None_type;
using vsip::impl::Type_list;
using vsip::impl::Make_type_list;

/// Provide an operation-specific backend list.
template <typename O> struct List;

template <typename O,                               // operation
          typename S = typename Signature<O>::type, // signature
          typename L = typename List<O>::type,      // list of backends
          typename B = typename L::first,           // current backend
          typename N = typename L::rest,            // the remainder of the list
          typename E = Evaluator<O, B, S>,          // the current evaluator
          bool V = E::ct_valid>
struct Dispatcher;

/// If ct-check fails, move to next in the list.
template <typename O, typename S, typename L, typename B, typename N, typename E>
struct Dispatcher<O, S, L, B, N, E, false> : Dispatcher<O, S, N> {};

template <typename O, typename S>
struct No_backend_available_for;

template <typename O, typename S, typename L, typename B, typename E>
struct Dispatcher<O, S, L, B, None_type, E, false>
  : No_backend_available_for<O, S> 
{};

/// Specialization for compile-time only checking. Instead of a full signature,
/// it takes a type parameter directly. Thus it would be accessed as
/// e.g 'Dispatcher<Operation, float>::type'.
template <typename O,
          typename T,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, T, L, B, N, E, true>
{
  typedef B backend;
  typedef typename E::backend_type type;
};

/// General case for R(A).
template <typename O,
          typename R, typename A,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A), L, B, N, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A), B>::type profiler_type;

  static R dispatch(A a)
  {
    if (E::rt_valid(a))
    {
      profiler_type profiler(a);
      return E::exec(a);
    }
    else return Dispatcher<O, R(A), N>::dispatch(a);
  }
};

/// Terminator for R(A): If rt-check fails, give up.
template <typename O, typename R, typename A, typename L, typename B, typename E>
struct Dispatcher<O, R(A), L, B, None_type, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A), B>::type profiler_type;

  static R dispatch(A a)
  {
    if (E::rt_valid(a))
    {
      profiler_type profiler(a);
      return E::exec(a);
    }
    else VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// Terminator for R(A): Give up.
template <typename O, typename R, typename A, typename L, typename B, typename E>
struct Dispatcher<O, R(A), L, B, None_type, E, false>
{
  static R dispatch(A)
  {
    VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// General case for R(A1, A2).
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2), L, B, N, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2), B>::type profiler_type;

  static R dispatch(A1 a1, A2 a2)
  {
    if (E::rt_valid(a1, a2))
    {
      profiler_type profiler(a1, a2);
      return E::exec(a1, a2);
    }
    else return Dispatcher<O, R(A1, A2), N>::dispatch(a1, a2);
  }
};

/// Terminator for R(A1, A2): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2), L, B, None_type, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2), B>::type profiler_type;

  static R dispatch(A1 a1, A2 a2)
  {
    if (E::rt_valid(a1, a2))
    {
      profiler_type profiler(a1, a2);
      return E::exec(a1, a2);
    }
    else VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2): Give up.
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2), L, B, None_type, E, false>
{
  static R dispatch(A1, A2)
  {
    VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// General case for R(A1, A2, A3).
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2, A3), L, B, N, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3), B>::type profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3)
  {
    if (E::rt_valid(a1, a2, a3))
    {
      profiler_type profiler(a1, a2, a3);
      return E::exec(a1, a2, a3);
    }
    else return Dispatcher<O, R(A1, A2, A3), N>::dispatch(a1, a2, a3);
  }
};

/// Terminator for R(A1, A2, A3): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3), L, B, None_type, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3), B>::type profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3)
  {
    if (E::rt_valid(a1, a2, a3))
    {
      profiler_type profiler(a1, a2, a3);
      return E::exec(a1, a2, a3);
    }
    else VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3), L, B, None_type, E, false>
{
  static R dispatch(A1, A2, A3)
  {
    VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// General case for R(A1, A2, A3, A4).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4), L, B, N, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3, A4), B>::type
    profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    if (E::rt_valid(a1, a2, a3, a4))
    {
      profiler_type profiler(a1, a2, a3, a4);
      return E::exec(a1, a2, a3, a4);
    }
    else return Dispatcher<O, R(A1, A2, A3, A4), N>::dispatch(a1, a2, a3, a4);
  }
};

/// Terminator for R(A1, A2, A3, A4): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4), L, B, None_type, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3, A4), B>::type
    profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    if (E::rt_valid(a1, a2, a3, a4))
    {
      profiler_type profiler(a1, a2, a3, a4);
      return E::exec(a1, a2, a3, a4);
    }
    else VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3, A4): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4), L, B, None_type, E, false>
{
  static R dispatch(A1, A2, A3, A4)
  {
    VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};


/// General case for R(A1, A2, A3, A4, A5).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5), L, B, N, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3, A4, A5), B>::type
    profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5))
    {
      profiler_type profiler(a1, a2, a3, a4, a5);
      return E::exec(a1, a2, a3, a4, a5);
    }
    else return Dispatcher<O, R(A1, A2, A3, A4, A5), N>::dispatch
      (a1, a2, a3, a4, a5);
  }
};

/// Terminator for R(A1, A2, A3, A4, A5): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5), L, B, None_type, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3, A4, A5), B>::type
    profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5))
    {
      profiler_type profiler(a1, a2, a3, a4, a5);
      return E::exec(a1, a2, a3, a4, a5);
    }
    else VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3, A4, A5): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5), L, B, None_type, E, false>
{
  static R dispatch(A1, A2, A3, A4, A5)
  {
    VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};


/// General case for R(A1, A2, A3, A4, A5, A6).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5, A6), L, B, N, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3, A4, A5, A6), B>::type
    profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5, a6))
    {
      profiler_type profiler(a1, a2, a3,a4, a5, a6);
      return E::exec(a1, a2, a3, a4, a5, a6);
    }
    else return Dispatcher<O, R(A1, A2, A3, A4, A5, A6), N>::dispatch
      (a1, a2, a3, a4, a5, a6);
  }
};

/// Terminator for R(A1, A2, A3, A4, A5, A6): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5, A6), L, B, None_type, E, true>
{
  typedef B backend;
  typedef typename Profile_policy_selector<O, R(A1, A2, A3, A4, A5, A6), B>::type
    profiler_type;

  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5, a6))
    {
      profiler_type profiler(a1, a2, a3, a4, a5, a6);
      return E::exec(a1, a2, a3, a4, a5, a6);
    }
    else VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3, A4, A5, A6): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5, A6), L, B, None_type, E, false>
{
  static R dispatch(A1, A2, A3, A4, A5, A6)
  {
    VSIP_IMPL_THROW(impl::unimplemented("No backend"));
  }
};

} // namespace vsip_csl::dispatcher

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


} // namespace vsip_csl

#endif
