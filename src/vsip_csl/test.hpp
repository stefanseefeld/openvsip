/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/test.hpp
    @author  Jules Bergmann
    @date    2005-01-25
    @brief   VSIPL++ CodeSourcery Library: Common declarations and 
             definitions for testing.
*/

#ifndef VSIP_CSL_TEST_HPP
#define VSIP_CSL_TEST_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdlib>
#include <cassert>

#include <vsip/support.hpp>
#include <vsip/complex.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>


namespace vsip_csl
{

/***********************************************************************
  Definitions
***********************************************************************/

// Define level of testing
//   0 - low-level (avoid long-running and long-compiling tests)
//   1 - default
//   2 - high-level (enable long-running tests)

#ifndef VSIP_IMPL_TEST_LEVEL
#  define VSIP_IMPL_TEST_LEVEL 1
#endif

// Run tests for double precision
#ifndef VSIP_IMPL_TEST_DOUBLE
  // PAS doesn't support double
#  if VSIP_IMPL_PAR_SERVICE == 2
#    define VSIP_IMPL_TEST_DOUBLE 0
#  else
#    define VSIP_IMPL_TEST_DOUBLE 1
#  endif
#endif

// Run tests for long-double precision
#ifndef VSIP_IMPL_TEST_LONG_DOUBLE
// PAS doesn't support long-double
#  if VSIP_IMPL_PAR_SERVICE == 2
#    define VSIP_IMPL_TEST_LONG_DOUBLE 0
#  else
#    define VSIP_IMPL_TEST_LONG_DOUBLE 1
#  endif
#endif


/// Compare two floating-point values for equality.
///
/// Algorithm from:
///    www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm

template <typename T>
struct Almost_equal
{
  static bool eq(
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
};

template <typename T>
struct Almost_equal<std::complex<T> >
{
  static bool eq(
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
};



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

template <typename T, typename Block1, typename Block2>
inline bool
view_equal(vsip::const_Tensor<T, Block1> v, vsip::const_Tensor<T, Block2> w)
{
  if (v.size(0) != w.size(0) ||
      v.size(1) != w.size(1) ||
      v.size(2) != w.size(2)) return false;
  for (vsip::length_type i = 0; i != v.size(0); ++i)
    for (vsip::length_type j = 0; j != v.size(1); ++j)
      for (vsip::length_type k = 0; k != v.size(2); ++k)
	if (!equal(v.get(i, j, k), w.get(i, j, k)))
	  return false;
  return true;
}


/// Compare two values for equality.
///
/// The following instantiations convert Lvalue_proxy variables to
/// the appropriate non-Lvalue versions and dispatch to the above
/// functions.
template <typename             T,
	  typename             Block,
	  vsip::dimension_type Dim>
inline bool
equal(
  vsip::impl::Lvalue_proxy<T, Block, Dim> const& val1, 
  T                                              val2)
{
  return equal(static_cast<T>(val1), val2);
}

template <typename             T,
	  typename             Block,
	  vsip::dimension_type Dim>
inline bool
equal(
  T                                              val1,
  vsip::impl::Lvalue_proxy<T, Block, Dim> const& val2) 
{
  return equal(val1, static_cast<T>(val2));
}

template <typename             T,
	  typename             Block1,
	  typename             Block2,
	  vsip::dimension_type Dim1,
	  vsip::dimension_type Dim2>
inline bool
equal(
  vsip::impl::Lvalue_proxy<T, Block1, Dim1> const& val1,
  vsip::impl::Lvalue_proxy<T, Block2, Dim2> const& val2)
{
  return equal(static_cast<T>(val1), static_cast<T>(val2));
}

template <typename             T,
	  typename             Block,
	  vsip::dimension_type Dim>
inline bool
equal(
  vsip::impl::Lvalue_proxy<T, Block, Dim> const& val1,
  vsip::impl::Lvalue_proxy<T, Block, Dim> const& val2)
{
  return equal(static_cast<T>(val1), static_cast<T>(val2));
}


/// Use a variable.  Useful for tests that must create a variable but
/// do not otherwise use it.

template <typename T>
inline void
use_variable(T const& /*t*/)
{
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
# define TEST_ASSERT_FUNCTION    ((const char *) 0)
#endif

#ifdef __STDC__
#  define __TEST_STRING(e) #e
#else
#  define __TEST_STRING(e) "e"
#endif

#define test_assert(expr)						\
  (static_cast<void>((expr) ? 0 :					\
		     (vsip_csl::test_assert_fail(__TEST_STRING(expr), __FILE__, __LINE__, \
				       TEST_ASSERT_FUNCTION), 0)))

} // namespace vsip_csl

#endif // VSIP_CSL_TEST_HPP
