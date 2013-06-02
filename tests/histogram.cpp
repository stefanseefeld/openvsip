//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T>
void
test_vector_histogram( length_type size )
{
  Vector<float> tmp(size);
  Rand<float> rgen(0);
  tmp = rgen.randu(size) * 8;

  Vector<T> v(size);
  for ( index_type i = 0; i < size; ++i )
    v.put(i, static_cast<T>(tmp.get(i)) );

  // with even bin sizes
  {
    Histogram<const_Vector, T> h(2, 6, 4);

    Vector<scalar_i> q(4);
    q = h(v);

    for ( index_type b = 0; b < 4; ++b )
    {
      scalar_i count = 0;
      for ( index_type i = 0; i < size; ++i )
        if ( (T(b * 2) <= v(i)) && (v(i) < T((b + 1) * 2)) )
          ++count;
      test_assert( q(b) == count );
    }
  }

  // with all values within specified range
  {
    Histogram<const_Vector, T> h(0, 8, 10);

    Vector<scalar_i> q(10);
    q = h(v);

    for ( index_type b = 0; b < 8; ++b )
    {
      scalar_i count = 0;
      for ( index_type i = 0; i < size; ++i )
        if ( (T(b) <= v(i)) && (v(i) < T(b + 1)) )
          ++count;
      test_assert( q(b+1) == count );
    }
    test_assert( q(0) == 0 );
    test_assert( q(9) == 0 );

    // verify it can accumulate results
    Vector<scalar_i> q2(10);
    q2 = h(v, true);

    for ( index_type b = 0; b < 8; ++b )
      test_assert( q2(b+1) == 2 * q(b+1) );
    test_assert( q2(0) == 0 );
    test_assert( q2(9) == 0 );
  }
}


template <typename T>
void
test_matrix_histogram( length_type rows, length_type cols )
{
  Matrix<float> tmp(rows, cols);
  Rand<float> rgen(0);
  tmp = rgen.randu(rows, cols) * 8.0;

  Matrix<T> m(rows, cols);
  for ( index_type i = 0; i < rows; ++i )
    for ( index_type j = 0; j < cols; ++j )
      m.put(i, j, static_cast<T>(tmp.get(i, j)));

  // with even bin sizes
  {
    Histogram<const_Matrix, T> h(2, 6, 4);

    Vector<scalar_i> q(4);
    q = h(m);

    for ( index_type b = 0; b < 4; ++b )
    {
      scalar_i count = 0;
      for ( index_type i = 0; i < rows; ++i )
        for ( index_type j = 0; j < cols; ++j )
          if ( (T(b * 2) <= m(i, j)) && (m(i, j) < T((b + 1) * 2)) )
            ++count;
      test_assert( q(b) == count );
    }
  }

  // with all values within specified range
  {
    Histogram<const_Matrix, T> h(0, 8, 10);

    Vector<scalar_i> q(10);
    q = h(m);

    for ( index_type b = 0; b < 8; ++b )
    {
      scalar_i count = 0;
      for ( index_type i = 0; i < rows; ++i )
        for ( index_type j = 0; j < cols; ++j )
          if ( (T(b) <= m(i, j)) && (m(i, j) < T(b + 1)) )
            ++count;
      test_assert( q(b+1) == count );
    }
    test_assert( q(0) == 0 );
    test_assert( q(9) == 0 );

    // verify it can accumulate results
    Vector<scalar_i> q2(10);
    q2 = h(m, true);

    for ( index_type b = 0; b < 8; ++b )
      test_assert( q2(b+1) == 2 * q(b+1) );
    test_assert( q2(0) == 0 );
    test_assert( q2(9) == 0 );
  }
}



template <typename T>
void
cases_by_type()
{
  test_vector_histogram<T>( 1024 );
  test_matrix_histogram<T>( 32, 32 );
}
  



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  cases_by_type<float>();
  cases_by_type<int>();
  cases_by_type<long>();

#if VSIP_IMPL_TEST_DOUBLE
  cases_by_type<double>();
#endif
#if VSIP_IMPL_TEST_LONG_DOUBLE
  cases_by_type<long double>();
#endif

  return EXIT_SUCCESS;
}
