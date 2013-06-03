//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef matvec_prod_hpp_
#define matvec_prod_hpp_

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/vector.hpp>
#include <test.hpp>

template <typename T0,
	  typename T1,
          typename T2,
          typename Block0,
          typename Block1,
          typename Block2>
void
check_prod(vsip::Matrix<T0, Block0> test,
	   vsip::Matrix<T1, Block1> chk,
	   vsip::Matrix<T2, Block2> gauge,
	   float                    threshold = 10.0)
{
  using namespace ovxx;
  typedef typename vsip::Promotion<T0, T1>::type return_type;
  typedef typename scalar_of<return_type>::type scalar_type;

  vsip::Index<2> idx;
  scalar_type err = vsip::maxval(((mag(chk - test)
			     / test::precision<scalar_type>::eps)
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
check_prod(vsip::Vector<T0, Block0> test,
	   vsip::Vector<T1, Block1> chk,
	   vsip::Vector<T2, Block2> gauge)
{
  using namespace ovxx;
  typedef typename vsip::Promotion<T0, T1>::type return_type;
  typedef typename scalar_of<return_type>::type scalar_type;

  vsip::Index<1> idx;
  scalar_type err = vsip::maxval(((mag(chk - test)
			     / test::precision<scalar_type>::eps)
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

#endif
