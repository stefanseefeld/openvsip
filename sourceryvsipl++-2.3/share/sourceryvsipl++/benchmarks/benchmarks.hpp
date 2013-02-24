/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/benchmarks.hpp
    @author  Don McCoy
    @date    2006-03-16
    @brief   VSIPL++ Library: Benchmark common definitions

*/

#ifndef VSIP_IMPL_BENCHMARKS_HPP
#define VSIP_IMPL_BENCHMARKS_HPP

#include "loop.hpp"

#ifdef VSIP_IMPL_SOURCERY_VPP

// Sourcery VSIPL++ provides certain resources such as system 
// timers that are needed for running the benchmarks.

#include <vsip/core/profile.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/test.hpp>

using vsip_csl::equal;

#else

// When linking with non-Sourcery versions of the lib, the
// definitions below provide a minimal set of these resources.

#include <time.h>

#include <cstdlib>
#include <cassert>

#include <vsip/support.hpp>
#include <vsip/complex.hpp>
#include <vsip/math.hpp>


#undef  VSIP_IMPL_NOINLINE
#define VSIP_IMPL_NOINLINE


namespace vsip
{
namespace impl
{
namespace profile
{

struct Posix_time
{
  static bool const valid = true; 
  static char* name() { return "Posix_time"; }
  static void init() { clocks_per_sec = CLOCKS_PER_SEC; }

  typedef clock_t stamp_type;
  static void sample(stamp_type& time) { time = clock(); }
  static stamp_type zero() { return stamp_type(); }
  static stamp_type f_clocks_per_sec() { return CLOCKS_PER_SEC; }
  static stamp_type add(stamp_type A, stamp_type B) { return A + B; }
  static stamp_type sub(stamp_type A, stamp_type B) { return A - B; }
  static float seconds(stamp_type time) { return (float)time / CLOCKS_PER_SEC; }
  static unsigned long ticks(stamp_type time) { return (unsigned long)time; }

  static stamp_type clocks_per_sec;
};


/// Timer class that keeps start/stop times.
///
/// Requires:
///   TP is a timer policy.

template <typename TP>
class P_timer {
private:
  typedef typename TP::stamp_type stamp_type;

  stamp_type	start_;
  stamp_type	stop_;

public:
  P_timer() {}

  void start() { TP::sample(start_); }
  void stop()  { TP::sample(stop_);  }

  stamp_type raw_delta() { return TP::sub(stop_, start_); }
  float delta() { return TP::seconds(TP::sub(stop_, start_)); }
};



/// Timer class that accumulates across multiple start/stop times.
///
/// Requires:
///   TP is a timer policy.

template <typename TP>
class P_acc_timer {
private:
  typedef typename TP::stamp_type stamp_type;

  stamp_type	total_;
  stamp_type	start_;
  stamp_type	stop_;
  unsigned	count_;

public:
  P_acc_timer() { total_ = stamp_type(); count_ = 0; }

  void start() { TP::sample(start_); }
  void stop()
  {
    TP::sample(stop_);
    total_ = TP::add(total_, TP::sub(stop_, start_));
    count_ += 1;
  }

  stamp_type raw_delta() const { return TP::sub(stop_, start_); }
  float delta() const { return TP::seconds(TP::sub(stop_, start_)); }
  float total() const { return TP::seconds(total_); }
  int   count() const { return count_; }
};

typedef Posix_time       DefaultTime;

typedef P_timer<DefaultTime>     Timer;
typedef P_acc_timer<DefaultTime> Acc_timer;


} // namespace vsip::impl::profile
} // namespace vsip::impl
} // namespace vsip


template <typename T>
struct Ops_info
{
  static unsigned int const div = 1;
  static unsigned int const sqr = 1;
  static unsigned int const mul = 1;
  static unsigned int const add = 1;
  static unsigned int const mag = 1;
};

template <typename T>
struct Ops_info<std::complex<T> >
{
  static unsigned int const div = 6 + 3 + 2; // mul + add + div
  static unsigned int const sqr = 2 + 1;     // mul + add
  static unsigned int const mul = 4 + 2;     // mul + add
  static unsigned int const add = 2;
  static unsigned int const mag = 2 + 1 + 1; // 2*mul + add + sqroot
};



/// Compare two floating-point values for equality.
///
/// Algorithm from:
///    www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm

template <typename T>
bool
almost_equal(
  T	A,
  T	B,
  T	rel_epsilon = 1e-4,
  T	abs_epsilon = 1e-6)
{
  if (vsip::mag(A - B) < abs_epsilon)
    return true;

  T relative_error;

  if (vsip::mag(B) > vsip::mag(A))
    relative_error = vsip::mag((A - B) / B);
  else
    relative_error = vsip::mag((B - A) / A);

  return (relative_error <= rel_epsilon);
}



template <typename T>
bool
almost_equal(
  std::complex<T>	A,
  std::complex<T>	B,
  T	rel_epsilon = 1e-4,
  T	abs_epsilon = 1e-6)
{
  if (vsip::mag(A - B) < abs_epsilon)
    return true;

  T relative_error;

  if (vsip::mag(B) > vsip::mag(A))
    relative_error = vsip::mag((A - B) / B);
  else
    relative_error = vsip::mag((B - A) / A);

  return (relative_error <= rel_epsilon);
}



/// Compare two values for equality.
template <typename T>
inline bool
equal(T val1, T val2)
{
  return val1 == val2;
}


/// Compare two floating point values for equality within epsilon.
///
/// Note: A fixed epsilon is not adequate for comparing the results
///       of all floating point computations.  Epsilon should be choosen 
///       based on the dynamic range of the computation.
template <>
inline bool
equal(float val1, float val2)
{
  return almost_equal<float>(val1, val2);
}



/// Compare two floating point (double) values for equality within epsilon.
template <>
inline bool
equal(double val1, double val2)
{
  return almost_equal<double>(val1, val2);
}



/// Compare two complex values for equality within epsilon.

template <typename T>
inline bool
equal(vsip::complex<T> val1, vsip::complex<T> val2)
{
  return equal(val1.real(), val2.real()) &&
         equal(val1.imag(), val2.imag());
}

template <typename T, typename Block1, typename Block2>
inline bool
view_equal(vsip::const_Vector<T, Block1> v, vsip::const_Vector<T, Block2> w)
{
  if (v.size() != w.size()) return false;
  for (vsip::length_type i = 0; i != v.size(); ++i)
    if (!equal(v.get(i), w.get(i)))
      return false;
  return true;
}

template <typename T, typename Block1, typename Block2>
inline bool
view_equal(vsip::const_Matrix<T, Block1> v, vsip::const_Matrix<T, Block2> w)
{
  if (v.size(0) != w.size(0) || v.size(1) != w.size(1)) return false;
  for (vsip::length_type i = 0; i != v.size(0); ++i)
    for (vsip::length_type j = 0; j != v.size(1); ++j)
      if (!equal(v.get(i, j), w.get(i, j)))
	return false;
  return true;
}



void inline
test_assert_fail(
  const char*  assertion,
  const char*  file,
  unsigned int line,
  const char*  function)
{
  fprintf(stderr, "TEST ASSERT FAIL: %s %s %d %s\n",
	  assertion, file, line, function);
  abort();
}

#if defined(__GNU__)
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define TEST_ASSERT_FUNCTION    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define TEST_ASSERT_FUNCTION    __func__
#  else
#   define TEST_ASSERT_FUNCTION    ((__const char *) 0)
#  endif
# endif
#else
# define TEST_ASSERT_FUNCTION    ((__const char *) 0)
#endif

#ifdef __STDC__
#  define __TEST_STRING(e) #e
#else
#  define __TEST_STRING(e) "e"
#endif

#define test_assert(expr)						\
  (static_cast<void>((expr) ? 0 :					\
		     (test_assert_fail(__TEST_STRING(expr), __FILE__, __LINE__, \
				       TEST_ASSERT_FUNCTION), 0)))



#endif // not VSIP_IMPL_SOURCERY_VPP


#endif // VSIP_IMPL_BENCHMARKS_HPP
