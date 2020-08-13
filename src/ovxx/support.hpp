//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// [support] basic macros, types, exceptions, and support functions.

#ifndef ovxx_support_hpp_
#define ovxx_support_hpp_

#include <vsip/support.hpp>
#if !OVXX_HAS_EXCEPTIONS
#include <iostream>
#endif
#if OVXX_HAVE_LTTNG
extern "C"
{
# include <lttng/tracef.h>
}
#endif

/// The maximum number of dimensions supported by this implementation.
#define VSIP_MAX_DIMENSION 3

/// If the compiler provides a way to annotate functions with an
/// indication that they never return (except possibly by throwing an
/// exception), then VSIP_NORETURN is defined to that annotation,
/// otherwise it is defined to nothing.
/// GCC supports it, as does Green Hills, as well as Intel.
/// The latter defines __GNUC__, too.
#if __GNUC__ >= 2 || defined(__ghs__)
#  define OVXX_NORETURN __attribute__ ((__noreturn__))
#else
#  define OVXX_NORETURN
#endif

/// If the compiler provides a way to annotate functions with an
/// indication that they should not be inlined, then VSIP_IMPL_NOINLINE
/// is defined to that annotation, otherwise it is defined to
/// nothing. 
///
/// Greenhills does not use this attribute.
#if __GNUC__ >= 2
#  define OVXX_NOINLINE __attribute__ ((__noinline__))
#else
#  define OVXX_NOINLINE
#endif

/// If the compiler provides a way to annotate functions that every
/// call inside the function should be inlined, then VSIP_IMPL_FLATTEN
/// is defined to that annotation, otherwise it is defined to nothing.
#if __GNUC__ >= 4
#  define OVXX_FLATTEN __attribute__ ((__flatten__))
#else
#  define OVXX_FLATTEN
#endif

/// Loop vectorization pragmas.
#if __INTEL_COMPILER && !__ICL
#  define PRAGMA_IVDEP _Pragma("ivdep")
#  define PRAGMA_VECTOR_ALWAYS _Pragma("vector always")
#else
#  define PRAGMA_IVDEP
#  define PRAGMA_VECTOR_ALWAYS
#endif

#if VSIP_HAS_EXCEPTIONS
#  define OVXX_DO_THROW(x) throw x      ///< Wraps throw statements
#else
#  define OVXX_DO_THROW(x) ovxx::fatal_exception(__FILE__, __LINE__, x)
#endif

// An unmet precondition is a programming error.
// In certain contexts (such as scripting environments)
// it may be handled by an exception, and in others it
// may be ignored entirely (such as optimized 'release' builds).
//
// This macro may be pre-defined by users. The default 
// implementation just uses assert().
#ifndef OVXX_PRECONDITION
# define OVXX_PRECONDITION(c) assert(c)
#endif

// A violated invariant is treated as a programming error
// (i.e., a bug in the library), which may be checked for in
// non-optimized 'debug' builds.
#define OVXX_INVARIANT(i) assert(i)

// Executing unreachable code implies the program state is corrupted,
// which we only check for in non-optimized 'debug' builds.
#define OVXX_UNREACHABLE(msg) assert(!msg)

// Allow some simple execution tracing.
#ifndef OVXX_TRACE
# if OVXX_HAVE_LTTNG
#  define OVXX_TRACE(...) tracef("ovxx::" __VA_ARGS__)
# else
#  define OVXX_TRACE(...)
# endif
#endif

namespace ovxx
{
/// The ovxx namespace is a superset of the vsip namespace
using namespace vsip;

// Define convenience template classes row_major and col_major.  They
// are used by Dense template class to choose the default dimension
// order for a choosen dimension.

template <dimension_type D> struct row_major;
template <> struct row_major<1> { typedef row1_type type;};
template <> struct row_major<2> { typedef row2_type type;};
template <> struct row_major<3> { typedef row3_type type;};

template <dimension_type D> struct col_major;
template <> struct col_major<1> { typedef col1_type type;};
template <> struct col_major<2> { typedef col2_type type;};
template <> struct col_major<3> { typedef col3_type type;};

#if !OVXX_HAS_EXCEPTIONS
/// This function is called instead of throwing an exception
/// when OVXX_HAS_EXCEPTIONS is 0.
OVXX_NORETURN inline void fatal_exception(char const * file, unsigned int line,
                                          std::exception const &e)
{
  std::cerr << "OpenVSIP: at " << file << ':' << line << '\n';
  std::cerr << "OpenVSIP: fatal: " << e.what() << std::endl;
  std::abort();
}
#endif

/// Exception to mark unimplemented VSIPL++ features and functionality.

class unimplemented : public std::runtime_error
{
public:
  explicit unimplemented(const std::string& str) : runtime_error(str) {}
};

} // namespace ovxx

#endif
