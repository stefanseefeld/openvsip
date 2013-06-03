//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>
#include <vsip/selgen.hpp>
#include <vsip/random.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>
#include <test.hpp>
#include "common.hpp"

#define VERBOSE  0
#define DO_FULL  0

using namespace ovxx;

/// Solve a linear system with the Toeplitz solver.

template <typename T,
	  typename Block1,
	  typename Block2>
void
test_toepsol(return_mechanism_type rtm,
	     Vector<T, Block1>     a,
	     Vector<T, Block2>     b)
{
  typedef typename scalar_of<T>::type scalar_type;

  length_type size = a.size();

  Vector<T> y(size);
  Vector<T> x(size, T(99));

  Matrix<T> aa(size, size);

  aa.diag() = a(0);
  for (index_type i=1; i<size; ++i)
  {
    aa.diag(+i) = a(i);
    aa.diag(-i) = impl_conj<T>(a(i));
  }


  // Solve toeplize system
  if (rtm == by_reference)
    toepsol(a, b, y, x);
  else
    x = toepsol(a, b, y);


  // Check answer
  Vector<T>           chk(size);
  Vector<scalar_type> gauge(size);

  chk   = prod(aa, x);
  gauge = prod(mag(aa), mag(x));

  for (index_type i=0; i<gauge.size(0); ++i)
    if (!(gauge(i) > scalar_type()))
      gauge(i) = scalar_type(1);

  Index<1> idx;
  scalar_type err = maxval(((mag(chk - b) / gauge)
			    / test::precision<scalar_type>::eps
			     / size),
			   idx);

#if VERBOSE
  cout << "aa = " << endl << aa  << endl;
  cout << "a = " << endl << a  << endl;
  cout << "b = " << endl << b  << endl;
  cout << "x = " << endl << x  << endl;
  cout << "chk = " << endl << chk  << endl;
  cout << "err = " << err  << endl;
#endif

  test_assert(err < 5.0);
}



/// Test a simple toeplitz linear system with zeros outside of diagonal.

template <typename T>
void
test_toepsol_diag(return_mechanism_type rtm,
		  T                     value,
		  length_type           size)
{
  Vector<T> a(size, T());
  Vector<T> b(size);

  a = T(); a(0) = value;

  b = ramp(T(1), T(1), size);

  test_toepsol(rtm, a, b);
}



template <typename T>
struct Toepsol_traits
{
  static T value(index_type i)
  {
    if (i == 0) return T(5);
    if (i == 1) return T(0.5);
    if (i == 2) return T(0.2);
    if (i == 3) return T(0.1);
    return T(0);
  }
};



template <typename T>
struct Toepsol_traits<complex<T> >
{
  static complex<T> value(index_type i)
  {
    if (i == 0) return complex<T>(5, 0);
    if (i == 1) return complex<T>(0.5, .1);
    if (i == 2) return complex<T>(0.2, .1);
    if (i == 3) return complex<T>(0.1, .15);
    return complex<T>(0, 0);
  }
};



/// Test a general toeplitz linear system.

template <typename T>
void
test_toepsol_rand(return_mechanism_type rtm,
		  length_type           size,
		  length_type           loop)
{
  typedef typename scalar_of<T>::type scalar_type;

  Vector<T> a(size, T());
  Vector<T> b(size);

  a = T();

  for (index_type i=0; i<size; ++i)
    a(i) = Toepsol_traits<T>::value(i);

  Rand<T> rand(1);

  for (index_type l=0; l<loop; ++l)
  {
    b = rand.randu(size);
    test_toepsol(rtm, a, b);
  }
}



/// Test a general toeplitz linear system (with distributed views).

/// Test that toeplitz solver will correctly work when given
/// distributed views.  Solver is not parallel.

template <typename T,
	  typename MapT>
void
test_toepsol_dist(return_mechanism_type rtm,
		  length_type           size,
		  length_type           loop)
{
  typedef typename scalar_of<T>::type scalar_type;

  typedef Dense<1, T, row1_type, MapT> block_type;
  typedef Vector<T, block_type>        view_type;

  MapT map(num_processors());

  view_type a(size, T(), map);
  view_type b(size, map);

  a = T();

  for (index_type i=0; i<size; ++i)
    a(i) = Toepsol_traits<T>::value(i);

  Rand<T> rand(1);

  for (index_type l=0; l<loop; ++l)
  {
    parallel::assign_local(b, rand.randu(size)); // b = rand.randu(size);
    test_toepsol(rtm, a, b);
  }
}



/// Test a non positive-definite toeplitz linear system.

template <typename T>
void
test_toepsol_illformed(return_mechanism_type rtm,
		       length_type           size)
{
  typedef typename scalar_of<T>::type scalar_type;

  Vector<T> a(size, T());
  Vector<T> b(size);

  // Specify a non positive-definite matrix.
  a = T();
  a(0) = T(1);
  a(1) = T(1);

  Rand<T> rand(1);

  b = rand.randu(size);

  int pass = 0;
  try
  {
    test_toepsol(rtm, a, b);
    pass = 0;
  }
  catch (const std::exception& error)
  {
    if (error.what() == std::string("TOEPSOL: not full rank"))
      pass = 1;
  }

  test_assert(pass == 1);
}


void
toepsol_cases(return_mechanism_type rtm)
{
  test_toepsol_diag<float>           (rtm, 1.0, 5);
  test_toepsol_diag<complex<float> > (rtm, complex<float>(2.0, 0.0), 5);

  test_toepsol_rand<float>           (rtm, 4, 5);
  test_toepsol_rand<complex<float> > (rtm, 6, 5);

#if VSIP_IMPL_TEST_DOUBLE
  test_toepsol_diag<double>          (rtm, 2.0, 5);
  test_toepsol_diag<complex<double> >(rtm, complex<double>(3.0, 0.0), 5);
  test_toepsol_rand<double>          (rtm, 5, 5);
  test_toepsol_rand<complex<double> >(rtm, 7, 5);
#endif

#if VSIP_HAS_EXCEPTIONS
  test_toepsol_illformed<float>      (rtm, 4);
#endif
#if OVXX_PARALLEL
  test_toepsol_dist<float, Map<Block_dist> >(rtm, 4, 5);
#endif
}
  
int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  toepsol_cases(by_reference);
  toepsol_cases(by_value);
}
