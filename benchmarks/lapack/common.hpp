//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

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
