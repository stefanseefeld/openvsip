/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/test-prod.hpp
    @author  Jules Bergmann
    @date    2005-09-12
    @brief   VSIPL++ Library: Common definitions for matrix product tests.
*/

#ifndef VSIP_TESTS_TEST_PROD_HPP
#define VSIP_TESTS_TEST_PROD_HPP


/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/vector.hpp>

#include <vsip_csl/test-precision.hpp>
#include <vsip_csl/test.hpp>


#if VERBOSE
#include <iostream>
#endif


/***********************************************************************
  Reference Definitions
***********************************************************************/

template <typename T0,
	  typename T1,
          typename T2,
          typename Block0,
          typename Block1,
          typename Block2>
void
check_prod(
  vsip::Matrix<T0, Block0> test,
  vsip::Matrix<T1, Block1> chk,
  vsip::Matrix<T2, Block2> gauge,
  float                    threshold = 10.0)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;

  vsip::Index<2> idx;
  scalar_type err = vsip::maxval(((mag(chk - test)
			     / vsip_csl::Precision_traits<scalar_type>::eps)
			    / gauge),
			   idx);

#if VERBOSE
  if (err >= threshold)
  {
    std::cout << "test  =\n" << test;
    std::cout << "chk   =\n" << chk;
    std::cout << "gauge =\n" << gauge;
    std::cout << "err = " << err << std::endl;
  }
#endif

  test_assert(err < threshold);
}


template <typename T0,
	  typename T1,
          typename T2,
          typename Block0,
          typename Block1,
          typename Block2>
void
check_prod(
  vsip::Vector<T0, Block0> test,
  vsip::Vector<T1, Block1> chk,
  vsip::Vector<T2, Block2> gauge)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::scalar_of<return_type>::type scalar_type;

  vsip::Index<1> idx;
  scalar_type err = vsip::maxval(((mag(chk - test)
			     / vsip_csl::Precision_traits<scalar_type>::eps)
			    / gauge),
			   idx);

#if VERBOSE
  std::cout << "test  =\n" << test;
  std::cout << "chk   =\n" << chk;
  std::cout << "gauge =\n" << gauge;
  std::cout << "err = " << err << std::endl;
#endif

  test_assert(err < 10.0);
}


template <> float  vsip_csl::Precision_traits<float>::eps = 0.0;
template <> double vsip_csl::Precision_traits<double>::eps = 0.0;


#endif // VSIP_TESTS_TEST_PROD_HPP
