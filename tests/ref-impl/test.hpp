/***********************************************************************

  File:   test.hpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   08/22/2002

  Contents: Functions useful when testing code.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.

***********************************************************************/

#ifndef VSIP_TEST_HPP
#define VSIP_TEST_HPP

/***********************************************************************
  Notes
***********************************************************************/

/*
  These functions ease testing the implementation.
 */

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/matrix.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#ifdef UNIT_TEST_FAILURE_PRINT
#include <iostream>
#endif /* UNIT_TEST_FAILURE_PRINT */
#include <stdlib.h>

/***********************************************************************
  Macros
***********************************************************************/

#ifdef UNIT_TEST_FAILURE_PRINT
#define insist(predicate)	\
{				\
  if (!(predicate)) {		\
    std::cerr << "Failure in " << __FILE__ << ", line " << __LINE__ << ".\n";	\
    abort ();			\
  }				\
}
#else
#define insist(predicate)	\
{				\
  if (!(predicate)) {		\
    abort ();			\
  }				\
}
#endif /* UNIT_TEST_FAILURE_PRINT */


/***********************************************************************
  Function Declarations
***********************************************************************/

template <typename T>
inline bool
equal (T const& operand1,
       T const& operand2) VSIP_NOTHROW;
			/* Returns:
			     true iff the two operands are equal or
			     nearly equal.  */

/* Check particular values in views.  */

template <typename T,
          typename Block>
inline void
check_entry (vsip::const_Vector<T, Block> const& v,
	     vsip::index_type                 idx,
	     T const&                      answer) VSIP_NOTHROW;
		/* Requires:
		     IDX must be a valid index for V.
		   Effect:
		     Aborts if v.get (idx) != answer.  */

template <typename T,
          typename Block>
inline void
check_entry (const vsip::const_Matrix<T, Block>& m,
	     vsip::index_type                 idx0,
	     vsip::index_type                 idx1,
	     T const&                      answer) VSIP_NOTHROW;
		/* Requires:
		     (IDX0, IDX1) must be a valid for M.
		   Effect:
		     Aborts if m.get (idx0, idx1) != answer.  */

/***********************************************************************
  Inline Function Definitions
***********************************************************************/

template <typename T>
inline bool
equal (T const& operand1,
       T const& operand2) VSIP_NOTHROW
{
  /* This may fail when comparing floating point numbers.  If so,
     provide a more flexible implementation.  Such an implementation
     could compare the absolute value of the operands with
     'std::numeric_limits<T>::epsilon ()'.  Unfortunately, some
     platforms' compilers do not support <limits> or <cmath> so
     another approach would be needed.
  */
  return operand1 == operand2;
}

/* Consider any two floating point numbers to be equal if they are
   within 1.0e-06 of each other.  */
template <>
inline
bool
equal (vsip::scalar_f const& operand1,
       vsip::scalar_f const& operand2) VSIP_NOTHROW
{
  return vsip::mag (operand1 - operand2) < static_cast<vsip::scalar_f>(1.0e-4);
}

/* Consider any two complex point numbers to be equal if their
   absolute value is within 1.0e-06 of each other.  */
template <>
inline
bool
equal (vsip::cscalar_f const& operand1,
       vsip::cscalar_f const& operand2) VSIP_NOTHROW
{
  return vsip::mag (operand1 - operand2) < static_cast<vsip::scalar_f>(1.0e-6);
}

template <typename T,
          typename Block>
inline void
check_entry (vsip::const_Vector<T, Block> const& v,
	     vsip::index_type                 idx,
	     T const&                      answer) VSIP_NOTHROW
{
  insist (equal (v.get (idx), answer));
}

template <typename T,
          typename Block>
inline void
check_entry (const vsip::const_Matrix<T, Block>& m,
	     vsip::index_type                 idx0,
	     vsip::index_type                 idx1,
	     T const&                      answer) VSIP_NOTHROW
{
  insist (equal (m.get (idx0, idx1), answer));
}



template <typename T,
	  typename Block1,
	  typename Block2>
inline
float
maxdiff(
  vsip::const_Matrix<T, Block1> m1,
  vsip::const_Matrix<T, Block2> m2)
{
  vsip::Index<2> idx;
  float diff = vsip::maxval(mag(m1 - m2), idx);
  return diff;
}



template <typename T,
	  typename Block1,
	  typename Block2>
inline
float
maxdiff(
  vsip::const_Vector<T, Block1> v1,
  vsip::const_Vector<T, Block2> v2)
{
  vsip::Index<1> idx;
  float diff = vsip::maxval(mag(v1 - v2), idx);
  return diff;
}

#endif // VSIP_TEST_HPP
