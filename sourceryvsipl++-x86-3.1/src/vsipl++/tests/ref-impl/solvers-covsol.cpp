/***********************************************************************

  File:   solvers-covsol.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   02/16/2005

  Contents: Very simple test of the covsol solver.

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
  VSIPL++ Library

***********************************************************************/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/solvers.hpp>
#include <vsip/random.hpp>

#include "test.hpp"

using vsip::Matrix;
using vsip::index_type;
using vsip::cscalar_f;
using vsip::scalar_f;
using vsip::length_type;
using vsip::covsol;

/***********************************************************************
  Function Definitions
***********************************************************************/

template <typename T,
	  typename Block>
void
coherent(
  Matrix<T, Block>& data,
  vsip::Rand<T>& R,
  index_type b1,
  index_type b2,
  index_type k_start,
  index_type k_end,
  float   w_1,		// coherent weight on beam 1
  float   w_2, 		// coherent weight on beam 2
  float   w_diff)
{ 

  for (index_type k=k_start; k<k_end; ++k)
  {
    T val = R.randn();
    
    data.put(k, b1, data.get(k, b1) + w_1 * val + w_diff * R.randn());
    data.put(k, b2, data.get(k, b2) + w_2 * val + w_diff * R.randn());
  }
}

int
main (int argc, char** argv)
{

  vsip::vsipl v(argc, argv);
  vsip::Rand<cscalar_f> R(1);

  scalar_f w_background = 0.01f;

  length_type const NB = 3;
  length_type const NK = 1024;

  // scalar_f scale = 1.f*NK;

  Matrix<cscalar_f> data(NK, NB);
  Matrix<cscalar_f> copy(NK, NB);
  Matrix<cscalar_f> covar(NB, NB);
  Matrix<cscalar_f> B(NB, 1);
  Matrix<cscalar_f> X(NB, 1);
  Matrix<cscalar_f> X2(NB, 1);
  Matrix<cscalar_f> chk(NB, 1);

  // Put some data into a matrix with some covariance properties.
  if (true)
  {
    for (index_type b=0; b<NB; ++b)
      for (index_type k=0; k<NK; ++k)
	data.put(k, b, w_background * R.randn());
  }

  coherent(data, R, 0, 1, 0, NK, 0.20f, 1.00f, 0.001f);
  coherent(data, R, 2, 1, 0, NK, 0.10f, 1.00f, 0.001f);
  coherent(data, R, 0, 2, 0, NK, 0.50f, 0.25f, 0.001f);

  // Build covariance matrix.
  covar = prod(herm(data), data); // / scale;
  Matrix<cscalar_f> covar2 = prod(herm(data), data);

  assert(maxdiff(covar, covar2) < 1e-6);

  B = 0.f; B.put(1, 0, 1.f);
  // B /= scale;

  // Solve covariance system (copy data since covsol may overwrite it)
  copy = data; covsol(copy, B, X);
  copy = data; X2 = covsol(copy, B);

  assert(maxdiff(X, X2) < 1e-6);

  // Check solution.
  chk = prod(covar, X);

#ifndef NDEBUG
  vsip::scalar_f epsilon = 1e-3;
  vsip::scalar_f error = maxdiff(chk, B);

  assert(error < epsilon);
#endif
  
  return 0;
}
