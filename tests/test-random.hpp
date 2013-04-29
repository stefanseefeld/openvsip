//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef VSIP_TESTS_TEST_RANDOM_HPP
#define VSIP_TESTS_TEST_RANDOM_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#define USE_VPP_RANDOM 1

#include <vsip/support.hpp>
#include <vsip/complex.hpp>
#include <vsip/matrix.hpp>

#if USE_VPP_RANDOM
#  include <vsip/random.hpp>
#endif




/***********************************************************************
  Definitions
***********************************************************************/

/// Return a random value between -0.5 and +0.5

#if USE_VPP_RANDOM

/// Fill a matrix with random values.

template <typename T,
	  typename Block>
void
randm(vsip::Matrix<T, Block> m)
{
  vsip::Rand<T> rgen(1, true);

  m = rgen.randu(m.size(0), m.size(1)) - T(0.5);
}



/// Fill a vector with random values.

template <typename T,
	  typename Block>
void
randv(vsip::Vector<T, Block> v)
{
  vsip::Rand<T> rgen(1, true);

  v = rgen.randu(v.size()) - T(0.5);
}

#else // !USE_VPP_RANDOM

template <typename T>
struct Random
{
  static T value() { return T(1.f * rand()/(RAND_MAX+1.0)) - T(0.5); }
};

/// Specialization for random complex value.

template <typename T>
struct Random<vsip::complex<T> >
{
  static vsip::complex<T> value() {
    return vsip::complex<T>(Random<T>::value(), Random<T>::value());
  }
};



/// Fill a matrix with random values.

template <typename T,
	  typename Block>
void
randm(vsip::Matrix<T, Block> m)
{
  using vsip::index_type;

  for (index_type r=0; r<m.size(0); ++r)
    for (index_type c=0; c<m.size(1); ++c)
      m(r, c) = Random<T>::value();
}



/// Fill a vector with random values.

template <typename T,
	  typename Block>
void
randv(vsip::Vector<T, Block> v)
{
  using vsip::index_type;

  for (index_type i=0; i<v.size(0); ++i)
    v(i) = Random<T>::value();
}
#endif // USE_VPP_RANDOM

template <typename T, typename B>
void randomize(vsip::Vector<T, B> m) { randv(m);}
template <typename T, typename B>
void randomize(vsip::Matrix<T, B> m) { randm(m);}

#endif // VSIP_TESTS_TEST_RANDOM_HPP
