/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    mprod.cpp
    @author  Don McCoy
    @date    2008-05-08
    @brief   VSIPL++ Library: Simple demonstation of matrix products.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>

#include <vsip_csl/output.hpp>


/***********************************************************************
  Definitions
***********************************************************************/

using namespace vsip;
using namespace vsip_csl;


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  {  
    typedef vsip::scalar_f   T;
    Matrix<T> a(5, 4, T(2));
    Matrix<T> b(4, 3, T(3));
    Matrix<T> c(5, 3, T());
    
    c = prod(a, b);

    std::cout << "c = " << std::endl << c << std::endl;
  }

  {  
    typedef vsip::cscalar_f   T;
    Matrix<T> a(5, 4, T(2));
    Matrix<T> b(4, 3, T(3));
    Matrix<T> c(5, 3, T());
    
    c = prod(a, b);

    std::cout << "c = " << std::endl << c << std::endl;
  }

  return 0;
}
