//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dispatcher_dispatcher_hpp_
#define ovxx_dispatcher_dispatcher_hpp_

#include <ovxx/support.hpp>

namespace ovxx
{
namespace dispatcher
{
struct null_type;

template <typename H, typename T>
struct type_list
{
  typedef H head;
  typedef T tail;
};

template <typename H = null_type, typename...T>
struct make_type_list
{
  typedef type_list<H, typename make_type_list<T...>::type> type;
};

template<>
struct make_type_list<>
{
  typedef null_type type;
};

/// Define the operation-specific Evaluator signature.
template <typename O, typename R = void> 
struct Signature
{
  // The default signature is useful for a compile-time check only,
  // as there are no arguments to inspect at runtime.
  typedef R(type)();
};

/// Provide an operation-specific backend list.
template <typename O> struct List;

/// An Evaluator determines whether an Operation can be performed
/// with a particular backend.
///
/// Template parameters:
///   :O: Operation tag
///   :B: Backend tag
///   :S: Signature
///   :E: Enable (SFINAE) check.
template <typename O,
          typename B,
          typename S = typename Signature<O>::type,
	  typename E = void>
struct Evaluator
{
  static bool const ct_valid = false;
};

template <typename O,                               // operation
          typename S = typename Signature<O>::type, // signature
          typename L = typename List<O>::type,      // list of backends
          typename B = typename L::head,            // current backend
          typename N = typename L::tail,            // the remainder of the list
          typename E = Evaluator<O, B, S>,          // the current evaluator
          bool V = E::ct_valid>
struct Dispatcher;

/// If ct-check fails, move to next in the list.
template <typename O, typename S, typename L, typename B, typename N, typename E>
struct Dispatcher<O, S, L, B, N, E, false> : Dispatcher<O, S, N> {};

template <typename O, typename S> struct no_backend_available_for;

template <typename O, typename S, typename L, typename B, typename E>
struct Dispatcher<O, S, L, B, null_type, E, false> : no_backend_available_for<O, S> {};

/// Specialization for compile-time only checking. Instead of a full signature,
/// it takes a type parameter directly. Thus it would be accessed as
/// e.g 'Dispatcher<Operation, float>::type'.
template <typename O, typename T, typename L, typename B, typename N, typename E>
struct Dispatcher<O, T, L, B, N, E, true>
{
  typedef B backend;
  typedef typename E::backend_type type;
};

/// General case for R(A...).
template <typename O,
          typename L, typename B, typename N, typename E,
          typename R, typename ...A>
struct Dispatcher<O, R(A...), L, B, N, E, true>
{
  typedef B backend;
  static R dispatch(A... args)
  {
    if (E::rt_valid(args...))
      return E::exec(args...);
    else return Dispatcher<O, R(A...), N>::dispatch(args...);
  }
};

/// Terminator for R(A...): If rt-check fails, give up.
template <typename O, typename L, typename B, typename E, typename R, typename ...A>
struct Dispatcher<O, R(A...), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A... args)
  {
    if (E::rt_valid(args...)) return E::exec(args...);
    else throw std::runtime_error("No backend");
  }
};

/// Terminator for R(A...): Give up.
template <typename O, typename L, typename B, typename E, typename R, typename ...A>
struct Dispatcher<O, R(A... args), L, B, null_type, E, false>
{
  static R dispatch(A... args)
  {
    throw std::runtime_error("No backend");
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
