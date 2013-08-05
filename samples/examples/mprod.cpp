/* Copyright (c) 2008, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

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
