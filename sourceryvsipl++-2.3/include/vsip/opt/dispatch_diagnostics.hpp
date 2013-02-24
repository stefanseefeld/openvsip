/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/dispatch_diagnostics.hpp
    @author  Stefan Seefeld
    @date    2009-07-17
    @brief   VSIPL++ Library: Dispatch diagnostics harness.
*/

#ifndef VSIP_OPT_DISPATCH_DIAGNOSTICS_HPP
#define VSIP_OPT_DISPATCH_DIAGNOSTICS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/dispatch.hpp>
#include <vsip/core/dispatch_tags.hpp>
#include <iostream>
#include <iomanip>
#include <string>

namespace vsip_csl
{
namespace dispatcher
{

template <typename T> 
struct Backend_name
{
  static std::string name() { return "unknown";}
};

#define BE_NAME(T)	                    \
template <>				    \
struct Backend_name<be::T>                  \
{	                                    \
  static std::string name() { return ""#T;} \
};

#define BE_NAME_AS(T, N)                    \
template <>				    \
struct Backend_name<be::T>                  \
{			                    \
  static std::string name() { return ""#N; }\
};

BE_NAME(user)
BE_NAME(intel_ipp)
BE_NAME(transpose)
BE_NAME(mercury_sal)
BE_NAME(cbe_sdk)
BE_NAME(cuda)
BE_NAME(cml)
BE_NAME(dense_expr)
BE_NAME(copy)
BE_NAME(op_expr)
BE_NAME(simd_builtin)
BE_NAME(simd_loop_fusion)
BE_NAME_AS(simd_unaligned_loop_fusion, simd_ulf)
BE_NAME(fc_expr)
BE_NAME(rbo_expr)
BE_NAME(mdim_expr)
BE_NAME(loop_fusion)
BE_NAME(cvsip)
BE_NAME(opt)
BE_NAME(generic)

#undef BE_NAME_AS
#undef BE_NAME

/// Give access to Evaluator traits.
/// Template parameters:
///   :E: Evaluator
///   :S: Signature
template <typename E, typename S, bool Ct_valid = E::ct_valid>
struct Evaluator_traits;

template <typename E, typename A>
struct Evaluator_traits<E, void(A), true>
{
  static char const *name() { return E::name();}
  static bool rt_valid(A a) { return E::rt_valid(a);}
};

template <typename E, typename A>
struct Evaluator_traits<E, void(A), false>
{
  static char const *name() { return 0;}
  static bool rt_valid(A) { return false;}
};

template <typename E, typename A1, typename A2>
struct Evaluator_traits<E, void(A1, A2), true>
{
  static char const *name() { return E::name();}
  static bool rt_valid(A1 a1, A2 a2) { return E::rt_valid(a1, a2);}
};

template <typename E, typename A1, typename A2>
struct Evaluator_traits<E, void(A1, A2), false>
{
  static char const *name() { return 0;}
  static bool rt_valid(A1, A2) { return false;}
};

template <typename E, typename A1, typename A2, typename A3>
struct Evaluator_traits<E, void(A1, A2, A3), true>
{
  static char const *name() { return E::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3) { return E::rt_valid(a1, a2, a3);}
};

template <typename E, typename A1, typename A2, typename A3>
struct Evaluator_traits<E, void(A1, A2, A3), false>
{
  static char const *name() { return 0;}
  static bool rt_valid(A1, A2, A3) { return false;}
};

template <typename E, typename A1, typename A2, typename A3, typename A4>
struct Evaluator_traits<E, void(A1, A2, A3, A4), true>
{
  static char const *name() { return E::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3, A4 a4)
  { return E::rt_valid(a1, a2, a3, a4);}
};

template <typename E, typename A1, typename A2, typename A3, typename A4>
struct Evaluator_traits<E, void(A1, A2, A3, A4), false>
{
  static char const *name() { return 0;}
  static bool rt_valid(A1, A2, A3, A4) { return false;}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5>
struct Evaluator_traits<E, void(A1, A2, A3, A4, A5), true>
{
  static char const *name() { return E::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  { return E::rt_valid(a1, a2, a3, a4, a5);}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5>
struct Evaluator_traits<E, void(A1, A2, A3, A4, A5), false>
{
  static char const *name() { return 0;}
  static bool rt_valid(A1, A2, A3, A4, A5) { return false;}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
struct Evaluator_traits<E, void(A1, A2, A3, A4, A5, A6), true>
{
  static char const *name() { return E::name();}
  static bool rt_valid(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  { return E::rt_valid(a1, a2, a3, a4, a5, a6);}
};

template <typename E, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
struct Evaluator_traits<E, void(A1, A2, A3, A4, A5, A6), false>
{
  static char const *name() { return 0;}
  static bool rt_valid(A1, A2, A3, A4, A5, A6) { return false;}
};

/// Print diagnostics for the given evaluator.
/// Template parameters:
///   :B: backend
///   :E: evaluator
///   :A: argument
template <typename B, typename E, typename A>
void diagnostics(A a)
{
  typedef Evaluator_traits<E, void(A)> traits;
  bool rt_valid = traits::rt_valid(a);

  std::cout << "  - " << std::setw(20) << Backend_name<B>::name()
	    << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
	    << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) std::cout << "  (" << traits::name() << ")";
  std::cout << std::endl;
}

/// Print diagnostics for the given evaluator.
/// Template parameters:
///   :B: backend
///   :E: evaluator
///   :A1 - A2: arguments
template <typename B, typename E, typename A1, typename A2>
void diagnostics(A1 a1, A2 a2)
{
  typedef Evaluator_traits<E, void(A1, A2)> traits;
  bool rt_valid = traits::rt_valid(a1, a2);

  std::cout << "  - " << std::setw(20) << Backend_name<B>::name()
	    << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
	    << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) std::cout << "  (" << traits::name() << ")";
  std::cout << std::endl;
}

/// Print diagnostics for the given evaluator.
/// Template parameters:
///   :B: backend
///   :E: evaluator
///   :A1 - A3: arguments
template <typename B, typename E, typename A1, typename A2, typename A3>
void diagnostics(A1 a1, A2 a2, A3 a3)
{
  typedef Evaluator_traits<E, void(A1, A2, A3)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3);

  std::cout << "  - " << std::setw(20) << Backend_name<B>::name()
	    << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
	    << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) std::cout << "  (" << traits::name() << ")";
  std::cout << std::endl;
}

/// Print diagnostics for the given evaluator.
/// Template parameters:
///   :B: backend
///   :E: evaluator
///   :A1 - A4: arguments
template <typename B, typename E, typename A1, typename A2, typename A3, typename A4>
void diagnostics(A1 a1, A2 a2, A3 a3, A4 a4)
{
  typedef Evaluator_traits<E, void(A1, A2, A3, A4)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3, a4);

  std::cout << "  - " << std::setw(20) << Backend_name<B>::name()
	    << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
	    << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) std::cout << "  (" << traits::name() << ")";
  std::cout << std::endl;
}

/// Print diagnostics for the given evaluator.
/// Template parameters:
///   :B: backend
///   :E: evaluator
///   :A1 - A5: arguments
template <typename B, typename E,
	  typename A1, typename A2, typename A3, typename A4, typename A5>
void diagnostics(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
{
  typedef Evaluator_traits<E, void(A1, A2, A3, A4, A5)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3, a4, a5);

  std::cout << "  - " << std::setw(20) << Backend_name<B>::name()
	    << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
	    << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) std::cout << "  (" << traits::name() << ")";
  std::cout << std::endl;
}

/// Print diagnostics for the given evaluator.
/// Template parameters:
///   :B: backend
///   :E: evaluator
///   :A1 - A6: arguments
template <typename B, typename E,
	  typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
void diagnostics(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
{
  typedef Evaluator_traits<E, void(A1, A2, A3, A4, A5, A6)> traits;
  bool rt_valid = traits::rt_valid(a1, a2, a3, a4, a5, a6);

  std::cout << "  - " << std::setw(20) << Backend_name<B>::name()
	    << "  ct: " << std::setw(5) << (E::ct_valid ? "true" : "false")
	    << "  rt: " << std::setw(5) << (rt_valid ? "true" : "false");
  if (E::ct_valid) std::cout << "  (" << traits::name() << ")";
  std::cout << std::endl;
}

/// Do the same traversal as the Dispatcher, but don't skip ct_valid=false
template <typename O,                               // operation
          typename S = typename Signature<O>::type, // signature
          typename L = typename List<O>::type,      // list of backends
          typename B = typename L::first,           // current backend
          typename N = typename L::rest,            // the remainder of the list
          typename E = Evaluator<O, B, S> >         // the current evaluator
struct Diagnostics
{
  static bool const ct_valid = E::ct_valid || Diagnostics<O, S, N>::ct_valid;
};

/// Terminator for compile-time only Dispatcher.
template <typename O,
          typename T,
          typename L, typename B, typename E>
struct Diagnostics<O, T, L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
};

/// General case for R(A).
template <typename O,
          typename R, typename A,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid || Diagnostics<O, R(A), N>::ct_valid;
  static void diag(A a)
  {
    diagnostics<B, E, A>(a);
    Diagnostics<O, R(A), N>::diag(a);
  }
};

/// Terminator for R(A)
template <typename O, typename R, typename A, typename L, typename B, typename E>
struct Diagnostics<O, R(A), L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static void diag(A a) { diagnostics<B, E, A>(a);}
};

/// General case for R(A1, A2).
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2), N>::ct_valid;
  static void diag(A1 a1, A2 a2)
  {
    diagnostics<B, E, A1, A2>(a1, a2);
    Diagnostics<O, R(A1, A2), N>::diag(a1, a2);
  }
};

/// Terminator for R(A1, A2)
template <typename O,
          typename R, typename A1, typename A2,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2), L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static void diag(A1 a1, A2 a2) { diagnostics<B, E, A1, A2>(a1, a2);}
};

/// General case for R(A1, A2, A3).
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2, A3), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2, A3), N>::ct_valid;
  static void diag(A1 a1, A2 a2, A3 a3)
  {
    diagnostics<B, E, A1, A2, A3>(a1, a2, a3);
    Diagnostics<O, R(A1, A2, A3), N>::diag(a1, a2, a3);
  }
};

/// Terminator for R(A1, A2, A3).
template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3), L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static void diag(A1 a1, A2 a2, A3 a3) { diagnostics<B, E, A1, A2, A3>(a1, a2, a3);}
};

/// General case for R(A1, A2, A3, A4).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename N, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4), L, B, N, E>
{
  static bool const ct_valid = E::ct_valid ||
    Diagnostics<O, R(A1, A2, A3, A4), N>::ct_valid;
  static void diag(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    diagnostics<B, E, A1, A2, A3, A4>(a1, a2, a3, a4);
    Diagnostics<O, R(A1, A2, A3, A4), N>::diag(a1, a2, a3, a4);
  }
};

/// Terminator for R(A1, A2, A3, A4).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4), L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static void diag(A1 a1, A2 a2, A3 a3, A4 a4)
  { diagnostics<B, E, A1, A2, A3, A4>(a1, a2, a3, a4);}
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
  static void diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    diagnostics<B, E, A1, A2, A3, A4, A5>(a1, a2, a3, a4, a5);
    Diagnostics<O, R(A1, A2, A3, A4, A5), N>::diag(a1, a2, a3, a4, a5);
  }
};

/// Terminator for R(A1, A2, A3, A4, A5).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4, A5), L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static void diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  { diagnostics<B, E, A1, A2, A3, A4, A5>(a1, a2, a3, a4, a5);}
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
  static void diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    diagnostics<B, E, A1, A2, A3, A4, A5, A6>(a1, a2, a3, a4, a5, a6);
    Diagnostics<O, R(A1, A2, A3, A4, A5, A6), N>::diag(a1, a2, a3, a4, a5, a6);
  }
};

/// Terminator for R(A1, A2, A3, A4, A5, A6).
template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename A5, typename A6,
          typename L, typename B, typename E>
struct Diagnostics<O, R(A1, A2, A3, A4, A5, A6), L, B, None_type, E>
{
  static bool const ct_valid = E::ct_valid;
  static void diag(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  { diagnostics<B, E, A1, A2, A3, A4, A5, A6>(a1, a2, a3, a4, a5, a6);}
};

template <typename O, typename T>
struct is_operation_supported
{
  static bool const value = Diagnostics<O, T>::ct_valid;
};

} // namespace vsip_csl::dispatcher

/// print dispatch diagnostics for a given operation.
/// Template parameters:
///   :O: The operation (tag) to diagnose.
///   :A: The argument.
///
/// Where normally dispatch<O>(a) would be called to perform
/// an operation, dispatch_diagnostics<O>(a) can be used to
/// print out diagnostics without actually performing it.
template <typename O, typename A>
void dispatch_diagnostics(A a)
{
  typedef dispatcher::Diagnostics<O, void(A)> diagnostics;
  return diagnostics::diag(a);
}

template <typename O, typename A1, typename A2>
void dispatch_diagnostics(A1 a1, A2 a2)
{
  typedef dispatcher::Diagnostics<O, void(A1, A2)> diagnostics;
  return diagnostics::diag(a1, a2);
}

} // namespace vsip_csl

#endif
