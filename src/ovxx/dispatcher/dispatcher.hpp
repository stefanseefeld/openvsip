//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dispatcher_dispatcher_hpp_
#define ovxx_dispatcher_dispatcher_hpp_

#include <ovxx/support.hpp>
#include <ovxx/c++11.hpp>

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

// If we can use variadic templates, life is easy...
#if __GXX_EXPERIMENTAL_CXX0X__

template <typename H = null_type, typename...T>
struct make_type_list
{
  typedef type_list<H, typename make_type_list<T...>::type> type;
};

#else

template <typename T1 = null_type,
	  typename T2 = null_type,
	  typename T3 = null_type,
	  typename T4 = null_type,
	  typename T5 = null_type,
	  typename T6 = null_type,
	  typename T7 = null_type,
	  typename T8 = null_type,
	  typename T9 = null_type,
	  typename T10 = null_type,
	  typename T11 = null_type,
	  typename T12 = null_type,
	  typename T13 = null_type,
	  typename T14 = null_type,
	  typename T15 = null_type,
	  typename T16 = null_type,
	  typename T17 = null_type>
struct make_type_list
{
private:
  typedef typename 
  make_type_list<T2, T3, T4, T5, T6, T7, T8, T9, T10,
		 T11, T12, T13, T14, T15, T16, T17>::type Rest;
public:
  typedef type_list<T1, Rest> type;
};

#endif

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

// If we can use variadic templates, life is easy...
#ifdef __GXX_EXPERIMENTAL_CXX0X__

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

#else

/// General case for R(A).
template <typename O,
          typename R, typename A,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A), L, B, N, E, true>
{
  typedef B backend;
  static R dispatch(A a)
  {
    if (E::rt_valid(a)) return E::exec(a);
    else return Dispatcher<O, R(A), N>::dispatch(a);
  }
};

/// Terminator for R(A): If rt-check fails, give up.
template <typename O, typename R, typename A, typename L, typename B, typename E>
struct Dispatcher<O, R(A), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A a)
  {
    if (E::rt_valid(a))
      return E::exec(a);
    else OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// Terminator for R(A): Give up.
template <typename O, typename R, typename A, typename L, typename B, typename E>
struct Dispatcher<O, R(A), L, B, null_type, E, false>
{
  static R dispatch(A)
  {
    OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// General case for R(A1, A2).
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2), L, B, N, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2)
  {
    if (E::rt_valid(a1, a2)) return E::exec(a1, a2);
    else return Dispatcher<O, R(A1, A2), N>::dispatch(a1, a2);
  }
};

/// Terminator for R(A1, A2): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2)
  {
    if (E::rt_valid(a1, a2)) return E::exec(a1, a2);
    else OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2): Give up.
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2), L, B, null_type, E, false>
{
  static R dispatch(A1, A2)
  {
    OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// General case for R(A1, A2, A3).
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2, A3), L, B, N, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2, A3 a3)
  {
    if (E::rt_valid(a1, a2, a3)) return E::exec(a1, a2, a3);
    else return Dispatcher<O, R(A1, A2, A3), N>::dispatch(a1, a2, a3);
  }
};

/// Terminator for R(A1, A2, A3): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2, A3 a3)
  {
    if (E::rt_valid(a1, a2, a3)) return E::exec(a1, a2, a3);
    else OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3), L, B, null_type, E, false>
{
  static R dispatch(A1, A2, A3)
  {
    OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// General case for R(A1, A2, A3, A4).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename N, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4), L, B, N, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    if (E::rt_valid(a1, a2, a3, a4)) return E::exec(a1, a2, a3, a4);
    else return Dispatcher<O, R(A1, A2, A3, A4), N>::dispatch(a1, a2, a3, a4);
  }
};

/// Terminator for R(A1, A2, A3, A4): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    if (E::rt_valid(a1, a2, a3, a4)) return E::exec(a1, a2, a3, a4);
    else OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3, A4): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4), L, B, null_type, E, false>
{
  static R dispatch(A1, A2, A3, A4)
  {
    OVXX_DO_THROW(unimplemented("No backend"));
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
  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5)) return E::exec(a1, a2, a3, a4, a5);
    else return Dispatcher<O, R(A1, A2, A3, A4, A5), N>::dispatch
      (a1, a2, a3, a4, a5);
  }
};

/// Terminator for R(A1, A2, A3, A4, A5): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5)) return E::exec(a1, a2, a3, a4, a5);
    else OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3, A4, A5): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5), L, B, null_type, E, false>
{
  static R dispatch(A1, A2, A3, A4, A5)
  {
    OVXX_DO_THROW(unimplemented("No backend"));
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
  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5, a6)) return E::exec(a1, a2, a3, a4, a5, a6);
    else return Dispatcher<O, R(A1, A2, A3, A4, A5, A6), N>::dispatch
      (a1, a2, a3, a4, a5, a6);
  }
};

/// Terminator for R(A1, A2, A3, A4, A5, A6): If rt-check fails, give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5, A6), L, B, null_type, E, true>
{
  typedef B backend;
  static R dispatch(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    if (E::rt_valid(a1, a2, a3, a4, a5, a6)) return E::exec(a1, a2, a3, a4, a5, a6);
    else OVXX_DO_THROW(unimplemented("No backend"));
  }
};

/// Terminator for R(A1, A2, A3, A4, A5, A6): Give up.
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename E>
struct Dispatcher<O, R(A1, A2, A3, A4, A5, A6), L, B, null_type, E, false>
{
  static R dispatch(A1, A2, A3, A4, A5, A6)
  {
    OVXX_DO_THROW(unimplemented("No backend"));
  }
};

#endif

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
