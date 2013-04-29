/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Common functions used in lapack benchmarks.

#include <vsip/support.hpp>
#include <vsip/random.hpp>
#include <vsip/math.hpp>

/// Fill a matrix with random values between -1 and +1.
template <typename T,
	  typename Block>
void
randm(vsip::Matrix<T, Block> m)
{
  vsip::Rand<T> generator(0, 0);
  m = generator.randu(m.size(0), m.size(1));
  m = 2.f * m - 1.f;
}

/// Fill a positive-definite symmetric matrix with random values.
template <typename T,
	  typename Block>
void
randm_symm(vsip::Matrix<T, Block> m)
{
  randm(m);
  m = vsip::prodt(m, m);

  for (vsip::index_type r = 0; r < m.size(0); ++r)
    m(r, r) += .001;
}

// Fill a positive-definite hermitian matrix with random values
template <typename Block>
void
randm_symm(vsip::Matrix<std::complex<float>, Block> m)
{
  randm(m);
  m = vsip::prodh(m, m);

  for (vsip::index_type r = 0; r < m.size(0); ++r)
    m(r, r) += .001;
}
