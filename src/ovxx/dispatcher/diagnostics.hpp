//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dispatcher_diagnostics_hpp_
#define ovxx_dispatcher_diagnostics_hpp_

#include <ovxx/dispatcher/dispatcher.hpp>
#include <ovxx/dispatcher/tags.hpp>
#include <ovxx/detail/util.hpp>
#include <sstream>
#include <iomanip>
#include <string>

namespace ovxx
{
namespace dispatcher
{
// Convert a __PRETTY_FUNCTION__ string literal
// into the name of the evaluator this was called
// from.
inline std::string eval_name(char const *pretty_func)
{
  std::string name = pretty_func;
  // First remove traces of the calling function - we are
  // only interested in the outer scope name, i.e. "Evaluator<...>"
  if (name.find("static std::string ovxx::dispatcher::", 0) == 0)
    name = name.substr(37);
  else if (name.find("static char const *ovxx::dispatcher::", 0) == 0)
    name = name.substr(37);
  size_t i = name.find(">::name() [with", 0);
  if (i != std::string::npos)
    name = name.substr(0, i + 1) + name.substr(i + 9);
  // Remove redundant namespace prefixes
  while ((i = name.find("ovxx::dispatcher::")) != std::string::npos)
  {
    name = name.substr(0, i) + name.substr(i + 18);
  }
  // Remove template argument substitutions
  i = name.find(" [with", 0);
  if (i != std::string::npos)
    name = name.substr(0, i);
  return name;
}

#define OVXX_DISPATCH_EVAL_NAME eval_name(__PRETTY_FUNCTION__)

namespace detail
{
using ovxx::detail::yes_tag;
using ovxx::detail::no_tag;

// Detect presence of "char const *E::name()"
template <typename T, char const *(*)()>
struct ptsmf1_helper;

// Detect presence of "std::string E::name()"
template <typename T, std::string (*)()>
struct ptsmf2_helper;

template <typename T>
no_tag
has_name_helper(...);

template <typename T>
yes_tag
has_name_helper(int, ptsmf1_helper<T, &T::name>* p = 0);

template <typename T>
yes_tag
has_name_helper(int, ptsmf2_helper<T, &T::name>* p = 0);

template <typename E>
struct has_name
{
  static bool const value = 
  sizeof(has_name_helper<E>(0)) == sizeof(yes_tag);
};

// If E has a (static) 'name()' member function,
// use it as part of the diagnostic.
template <typename E, bool = has_name<E>::value>
struct evaluator_name
{
  static std::string name() { return "<no info>";}
};

template <typename E>
struct evaluator_name<E, true>
{
  static std::string name() { return E::name();}
};

template <typename T> 
struct backend_name
{
  static char const *name() { return "unknown backend";}
};

#define OVXX_BE_NAME(T)	                    \
template <>                                 \
struct backend_name<be::T>                  \
{                                           \
  static char const *name() { return ""#T;} \
};

OVXX_BE_NAME(user)
OVXX_BE_NAME(transpose)
OVXX_BE_NAME(opencl)
OVXX_BE_NAME(cuda)
OVXX_BE_NAME(blas)
OVXX_BE_NAME(dense_expr)
OVXX_BE_NAME(copy)
OVXX_BE_NAME(op_expr)
OVXX_BE_NAME(simd)
OVXX_BE_NAME(fc_expr)
OVXX_BE_NAME(rbo_expr)
OVXX_BE_NAME(mdim_expr)
OVXX_BE_NAME(loop_fusion)
OVXX_BE_NAME(cvsip)
OVXX_BE_NAME(opt)
OVXX_BE_NAME(generic)

#undef OVXX_BE_NAME

template <typename E>
struct backend_of 
{
  static char const *name() { return "unknown";}
};

template <typename O, typename B, typename S, typename E>
struct backend_of<Evaluator<O, B, S, E> >
{
  static char const *name() { return backend_name<B>::name();}
};

template <typename E, typename S, bool V = E::ct_valid>
struct traits;

// If we can use variadic templates, life is easy...
#if __GXX_EXPERIMENTAL_CXX0X__

template <typename E, typename ...A>
struct traits<E, void(A...), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A... args) { return E::rt_valid(args...);}
};

template <typename E, typename ...A>
struct traits<E, void(A...), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A...) { return false;}
};

#else

template <typename E, typename A>
struct traits<E, void(A), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A a) { return E::rt_valid(a);}
};

template <typename E, typename A>
struct traits<E, void(A), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A) { return false;}
};

template <typename E, typename A1, typename A2>
struct traits<E, void(A1, A2), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1 a1, A2 a2) { return E::rt_valid(a1, a2);}
};

template <typename E, typename A1, typename A2>
struct traits<E, void(A1, A2), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1, A2) { return false;}
};

template <typename E, typename A1, typename A2, typename A3>
struct traits<E, void(A1, A2, A3), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3) { return E::rt_valid(a1, a2, a3);}
};

template <typename E, typename A1, typename A2, typename A3>
struct traits<E, void(A1, A2, A3), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1, A2, A3) { return false;}
};

template <typename E, typename A1, typename A2, typename A3, typename A4>
struct traits<E, void(A1, A2, A3, A4), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3, A4 a4)
  { return E::rt_valid(a1, a2, a3, a4);}
};

template <typename E, typename A1, typename A2, typename A3, typename A4>
struct traits<E, void(A1, A2, A3, A4), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1, A2, A3, A4) { return false;}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5>
struct traits<E, void(A1, A2, A3, A4, A5), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  { return E::rt_valid(a1, a2, a3, a4, a5);}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5>
struct traits<E, void(A1, A2, A3, A4, A5), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1, A2, A3, A4, A5) { return false;}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
struct traits<E, void(A1, A2, A3, A4, A5, A6), true>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  { return E::rt_valid(a1, a2, a3, a4, a5, a6);}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
struct traits<E, void(A1, A2, A3, A4, A5, A6), false>
{
  static char const *name() { return backend_of<E>::name();}
  static bool rt_valid(A1, A2, A3, A4, A5, A6) { return false;}
};

#endif

template <typename E, typename S>
std::string backend_diag(bool rt_valid)
{
  typedef detail::traits<E, S> traits;
  std::ostringstream oss;
  oss << "  - " << std::setw(20) << traits::name()
      << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
      << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) oss << "  (" << evaluator_name<E>::name() << ")";
  oss << std::endl;
  return oss.str();
}
} // namespace ovxx::dispatcher::detail

// If we can use variadic templates, life is easy...
#if __GXX_EXPERIMENTAL_CXX0X__

template <typename E, typename ...A>
std::string backend_diag(A... args)
{
  typedef detail::traits<E, void(A...)> traits;
  bool rt_valid = traits::rt_valid(args...);
  return detail::backend_diag<E, void(A...)>(rt_valid);
}

#else

template <typename E, typename A>
std::string backend_diag(A a)
{
  typedef detail::traits<E, void(A)> traits;
  bool rt_valid = traits::rt_valid(a);
  return detail::backend_diag<E, void(A)>(rt_valid);
}

template <typename E, typename A1, typename A2>
std::string backend_diag(A1 a1, A2 a2)
{
  typedef detail::traits<E, void(A1, A2)> traits;
  bool rt_valid = traits::rt_valid(a1, a2);
  return detail::backend_diag<E, void(A1, A2)>(rt_valid);
}

template <typename E, typename A1, typename A2, typename A3>
std::string backend_diag(A1 a1, A2 a2, A3 a3)
{
  typedef detail::traits<E, void(A1, A2, A3)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3);
  return detail::backend_diag<E, void(A1, A2, A3)>(rt_valid);
}

template <typename E, typename A1, typename A2, typename A3, typename A4>
std::string backend_diag(A1 a1, A2 a2, A3 a3, A4 a4)
{
  typedef detail::traits<E, void(A1, A2, A3, A4)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3, a4);
  return detail::backend_diag<E, void(A1, A2, A3, A4)>(rt_valid);
}

template <typename B, typename E,
	  typename A1, typename A2, typename A3, typename A4, typename A5>
std::string backend_diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
{
  typedef detail::traits<E, void(A1, A2, A3, A4, A5)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3, a4, a5);
  return detail::backend_diag<E, void(A1, A2, A3, A4, A5)>(rt_valid);
}

template <typename B, typename E,
	  typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
std::string backend_diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
{
  typedef detail::traits<E, void(A1, A2, A3, A4, A5, A6)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3, a4, a5, a6);
  return detail::backend_diag<E, void(A1, A2, A3, A4, A5, A6)>(rt_valid);
}

#endif

/// Do the same traversal as the Dispatcher, but don't skip ct_valid=false
template <typename O,                               // operation
          typename S = typename Signature<O>::type, // signature
          typename L = typename List<O>::type,      // list of backends
          typename B = typename L::head,            // current backend
          typename N = typename L::tail,            // the remainder of the list
          typename E = Evaluator<O, B, S> >         // the current evaluator
struct Diagnostics
{
  static bool const ct_valid = E::ct_valid || Diagnostics<O, S, N>::ct_valid;
};

/// Terminator for compile-time only Dispatcher.
template <typename O,
          typename T,
          typename L, typename B, typename E>
struct Diagnostics<O, T, L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
};

// If we can use variadic templates, life is easy...
#if __GXX_EXPERIMENTAL_CXX0X__

template <typename O, typename L, typename B, typename N, typename E, typename R, typename ...A>
struct Diagnostics<O, R(A...), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid || Diagnostics<O, R(A...), N>::ct_valid;
  static std::string diag(A... args)
  {
    std::string msg = backend_diag<E, A...>(args...);
    msg += Diagnostics<O, R(A...), N>::diag(args...);
    return msg;
  }
};

/// Terminator
template <typename O, typename L, typename B, typename E, typename R, typename ...A>
struct Diagnostics<O, R(A...), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A... args) { return backend_diag<E, A...>(args...);}
};

#else

/// General case for R(A).
template <typename O,
          typename R, typename A,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid || Diagnostics<O, R(A), N>::ct_valid;
  static std::string diag(A a)
  {
    std::string msg = backend_diag<E, A>(a);
    msg += Diagnostics<O, R(A), N>::diag(a);
    return msg;
  }
};

/// Terminator for R(A)
template <typename O, typename R, typename A, typename L, typename B, typename E>
struct Diagnostics<O, R(A), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A a) { return backend_diag<E, A>(a);}
};

/// General case for R(A1, A2).
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2), N>::ct_valid;
  static std::string diag(A1 a1, A2 a2)
  {
    std::string msg = backend_diag<E, A1, A2>(a1, a2);
    msg += Diagnostics<O, R(A1, A2), N>::diag(a1, a2);
    return msg;
  }
};

/// Terminator for R(A1, A2)
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A1 a1, A2 a2) { return backend_diag<E, A1, A2>(a1, a2);}
};

/// General case for R(A1, A2, A3).
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2, A3), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2, A3), N>::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3)
  {
    std::string msg = backend_diag<E, A1, A2, A3>(a1, a2, a3);
    msg += Diagnostics<O, R(A1, A2, A3), N>::diag(a1, a2, a3);
    return msg;
  }
};

/// Terminator for R(A1, A2, A3).
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3) { return backend_diag<E, A1, A2, A3>(a1, a2, a3);}
};

/// General case for R(A1, A2, A3, A4).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2, A3, A4), N>::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    std::string msg = backend_diag<E, A1, A2, A3, A4>(a1, a2, a3, a4);
    msg += Diagnostics<O, R(A1, A2, A3, A4), N>::diag(a1, a2, a3, a4);
    return msg;
  }
};

/// Terminator for R(A1, A2, A3, A4).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3, A4 a4)
  { return backend_diag<E, A1, A2, A3, A4>(a1, a2, a3, a4);}
};

/// General case for R(A1, A2, A3, A4, A5).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4, A5), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2, A3, A4, A5), N>::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    std::string msg = backend_diag<E, A1, A2, A3, A4, A5>(a1, a2, a3, a4, a5);
    msg += Diagnostics<O, R(A1, A2, A3, A4, A5), N>::diag(a1, a2, a3, a4, a5);
    return msg;
  }
};

/// Terminator for R(A1, A2, A3, A4, A5).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4, A5), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  { return backend_diag<E, A1, A2, A3, A4, A5>(a1, a2, a3, a4, a5);}
};

/// General case for R(A1, A2, A3, A4, A5, A6).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4, A5, A6), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2, A3, A4, A5, A6), N>::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    std::string msg = backend_diag<E, A1, A2, A3, A4, A5, A6>(a1, a2, a3, a4, a5, a6);
    msg += Diagnostics<O, R(A1, A2, A3, A4, A5, A6), N>::diag(a1, a2, a3, a4, a5, a6);
    return msg;
  }
};

/// Terminator for R(A1, A2, A3, A4, A5, A6).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4, A5, A6), L, B, null_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static std::string diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  { return backend_diag<E, A1, A2, A3, A4, A5, A6>(a1, a2, a3, a4, a5, a6);}
};

#endif

template <typename O, typename S>
struct is_operation_supported
{
  static bool const value = Diagnostics<O, S>::ct_valid;
};

// If we can use variadic templates, life is easy...
#if __GXX_EXPERIMENTAL_CXX0X__

/// print dispatch diagnostics for a given operation.
/// Template parameters:
///   :O: The operation (tag) to diagnose.
///   :R: The return type.
///   :...A: The argument type(s).
///
/// Where normally dispatch<O>(args) would be called to perform
/// an operation, dispatch_diagnostics<O>(args) can be used to
/// print out diagnostics without actually performing it.
template <typename O, typename R, typename ...A>
std::string diagnostics(A... args)
{
  typedef Diagnostics<O, R(A...)> d;
  return d::diag(args...);
}

#else

/// print dispatch diagnostics for a given operation.
/// Template parameters:
///   :O: The operation (tag) to diagnose.
///   :R: The return type.
///   :A: The argument type.
///
/// Where normally dispatch<O>(a) would be called to perform
/// an operation, dispatch_diagnostics<O>(a) can be used to
/// print out diagnostics without actually performing it.
template <typename O, typename R, typename A>
std::string diagnostics(A a)
{
  typedef Diagnostics<O, R(A)> d;
  return d::diag(a);
}

template <typename O, typename R, typename A1, typename A2>
std::string diagnostics(A1 a1, A2 a2)
{
  typedef Diagnostics<O, R(A1, A2)> d;
  return d::diag(a1, a2);
}

template <typename O, typename R, typename A1, typename A2, typename A3>
std::string diagnostics(A1 a1, A2 a2, A3 a3)
{
  typedef Diagnostics<O, R(A1, A2, A3)> d;
  return d::diag(a1, a2, a3);
}

template <typename O, typename R, typename A1, typename A2, typename A3, typename A4>
std::string diagnostics(A1 a1, A2 a2, A3 a3, A4 a4)
{
  typedef Diagnostics<O, R(A1, A2, A3, A4)> d;
  return d::diag(a1, a2, a3, a4);
}

template <typename O, typename R, typename A1, typename A2, typename A3, typename A4, typename A5>
std::string diagnostics(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
{
  typedef Diagnostics<O, R(A1, A2, A3, A4, A5)> d;
  return d::diag(a1, a2, a3, a4, a5);
}

template <typename O, typename R, typename A1, typename A2, typename A3, typename A4,
	  typename A5, typename A6>
std::string diagnostics(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
{
  typedef Diagnostics<O, R(A1, A2, A3, A4, A5, A6)> d;
  return d::diag(a1, a2, a3, a4, a5, a6);
}

#endif

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
