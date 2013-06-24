//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef solvers_common_hpp_
#define solvers_common_hpp_

#include <vsip/support.hpp>
#include <vsip/complex.hpp>
#include <vsip/matrix.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T>
struct Test_traits
{
  static T offset() { return T(0);    }
  static T value1() { return T(2);    }
  static T value2() { return T(0.5);  }
  static T value3() { return T(-0.5); }
  static T conj(T a) { return a; }
  static bool is_positive(T a) { return (a > T(0)); }

  static vsip::mat_op_type const trans = vsip::mat_trans;
};

template <typename T>
struct Test_traits<vsip::complex<T> >
{
  static vsip::complex<T> offset() { return vsip::complex<T>(0, 2);      }
  static vsip::complex<T> value1() { return vsip::complex<T>(2);      }
  static vsip::complex<T> value2() { return vsip::complex<T>(0.5, 1); }
  static vsip::complex<T> value3() { return vsip::complex<T>(1, -1); }
  static vsip::complex<T> conj(vsip::complex<T> a) { return vsip::conj(a); }
  static bool is_positive(vsip::complex<T> a)
  { return (a.real() > T(0)) && (equal(a.imag(), T(0))); }

  static vsip::mat_op_type const trans = vsip::mat_herm;
};



template <typename T>
T tconj(T const& val)
{
  return Test_traits<T>::conj(val);
}

template <typename T>
bool
is_positive(T const& val)
{
  return Test_traits<T>::is_positive(val);
}



// Compute matrix-matrix produce C = A B

template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  vsip::const_Matrix<T, Block1> a,
  vsip::const_Matrix<T, Block2> b,
  vsip::Matrix      <T, Block3> c)
{
  using vsip::index_type;

  assert(a.size(0) == c.size(0));
  assert(b.size(1) == c.size(1));
  assert(a.size(1) == b.size(0));

  for (index_type i=0; i<c.size(0); ++i)
    for (index_type j=0; j<c.size(1); ++j)
    {
      T tmp = T();
      for (index_type k=0; k<a.size(1); ++k)
	tmp += a.get(i, k) * b.get(k, j);
      c(i, j) = tmp;
    }
}



// Check error between matrix-matrix product A B and expected C.

template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
float
prod_check(
  vsip::const_Matrix<T, Block1> a,
  vsip::const_Matrix<T, Block2> b,
  vsip::Matrix<T, Block3> c)
{
  using vsip::index_type;
  typedef typename ovxx::scalar_of<T>::type scalar_type;

  assert(a.size(0) == c.size(0));
  assert(b.size(1) == c.size(1));
  assert(a.size(1) == b.size(0));

  float err = 0.f;

  for (index_type i=0; i<c.size(0); ++i)
    for (index_type j=0; j<c.size(1); ++j)
    {
      T           tmp   = T();
      scalar_type guage = scalar_type();

      for (index_type k=0; k<a.size(1); ++k)
      {
	tmp   += a.get(i, k) * b.get(k, j);
	guage += vsip::mag(a.get(i, k)) * vsip::mag(b.get(k, j));
      }

      float err_ij = vsip::mag(tmp - c(i, j)) / 
        test::precision<scalar_type>::eps;
      if (guage > scalar_type())
	err_ij = err_ij/guage;
      err = std::max(err, err_ij);
    }

  return err;
}



// Compute matrix-matrix produce C = A' B

template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prodh(vsip::const_Matrix<T, Block1> a,
      vsip::const_Matrix<T, Block2> b,
      vsip::Matrix      <T, Block3> c)
{
  using vsip::index_type;

  assert(a.size(1) == c.size(0));
  assert(b.size(1) == c.size(1));
  assert(a.size(0) == b.size(0));

  for (index_type i=0; i<c.size(0); ++i)
    for (index_type j=0; j<c.size(1); ++j)
    {
      T tmp = T();
      for (index_type k=0; k<a.size(0); ++k)
	tmp += Test_traits<T>::conj(a.get(k, i)) * b.get(k, j);
      c(i, j) = tmp;
    }
}

#endif
