//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector magsq.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "../benchmark.hpp"

using namespace ovxx;


/***********************************************************************
  Definitions - vector element-wise multiply
***********************************************************************/

template <typename T>
struct t_vmagsq1 : Benchmark_base
{
  typedef typename scalar_of<T>::type scalar_type;

  char const* what() { return "t_vmagsq1"; }
  int ops_per_point(length_type)
  {
    if (is_complex<T>::value)
      return 2*ovxx::ops_count::traits<scalar_type>::mul + 
        ovxx::ops_count::traits<scalar_type>::add;
    else
      return ovxx::ops_count::traits<T>::mul;
  }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>           A(size, T());
    Vector<scalar_type> Z(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    A.put(0, T(3));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = magsq(A);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(Z.get(i), magsq(A.get(i))));
  }

  void diag()
  {
    length_type size = 256;
    Vector<T>           A(size, T());
    Vector<scalar_type> Z(size);

    std::cout << ovxx::assignment::diagnostics(Z, magsq(A)) << std::endl;
  }
};



template <typename T>
struct t_vmag1 : Benchmark_base
{
  typedef typename scalar_of<T>::type scalar_type;

  char const* what() { return "t_vmag1"; }
  int ops_per_point(length_type)
  {
    if (is_complex<T>::value)
      return 2*ovxx::ops_count::traits<scalar_type>::mul + 
        ovxx::ops_count::traits<scalar_type>::add;
    else
      return ovxx::ops_count::traits<T>::mul;
  }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>           A(size, T());
    Vector<scalar_type> Z(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    A.put(0, T(3));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = mag(A);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(Z.get(i), mag(A.get(i))));
  }

  void diag()
  {
    length_type size = 256;
    Vector<T>           A(size, T());
    Vector<scalar_type> Z(size);

    std::cout << ovxx::assignment::diagnostics(Z, mag(A)) << std::endl;
  }
};



template <typename T>
struct t_vmag_dense_mat : Benchmark_base
{
  typedef typename scalar_of<T>::type scalar_type;

  static length_type const rows = 2;

  char const* what() { return "t_vmag_dense_mat"; }
  int ops_per_point(length_type)
  {
    if (is_complex<T>::value)
      return rows * (2*ovxx::ops_count::traits<scalar_type>::mul + 
		     ovxx::ops_count::traits<scalar_type>::add);
    else
      return rows * ovxx::ops_count::traits<T>::mul;
  }
  int riob_per_point(length_type) { return rows*2*sizeof(T); }
  int wiob_per_point(length_type) { return rows*1*sizeof(T); }
  int mem_per_point(length_type)  { return rows*3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Matrix<T>           A(rows, size, T());
    Matrix<scalar_type> Z(rows, size);

    Rand<T> gen(0, 0);
    A = gen.randu(rows, size);

    A.put(0, 0, T(3));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = mag(A);
    time = t1.elapsed();
    
    for (index_type r=0; r<rows; ++r)
      for (index_type i=0; i<size; ++i)
	test_assert(equal(Z.get(r, i), mag(A.get(r, i))));
  }

  void diag()
  {
    length_type size = 256;
    Matrix<T>           A(rows, size, T());
    Matrix<scalar_type> Z(rows, size);

    std::cout << ovxx::assignment::diagnostics(Z, mag(A)) << std::endl;
  }
};



template <typename T>
struct t_vmag_nondense_mat : Benchmark_base
{
  typedef typename scalar_of<T>::type scalar_type;

  static length_type const rows = 2;

  char const* what() { return "t_vmag_nondense_mat"; }
  int ops_per_point(length_type)
  {
    if (is_complex<T>::value)
      return rows * (2*ovxx::ops_count::traits<scalar_type>::mul + 
		     ovxx::ops_count::traits<scalar_type>::add);
    else
      return rows * ovxx::ops_count::traits<T>::mul;
  }
  int riob_per_point(length_type) { return rows*2*sizeof(T); }
  int wiob_per_point(length_type) { return rows*1*sizeof(T); }
  int mem_per_point(length_type)  { return rows*3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Matrix<T>           Asup(rows, size + 16, T());
    Matrix<scalar_type> Zsup(rows, size + 16);

    typename Matrix<T>::subview_type A = Asup(Domain<2>(rows, size));
    typename Matrix<T>::subview_type Z = Zsup(Domain<2>(rows, size));

    Rand<T> gen(0, 0);
    A = gen.randu(rows, size);

    A.put(0, 0, T(3));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = mag(A);
    time = t1.elapsed();
    
    for (index_type r=0; r<rows; ++r)
      for (index_type i=0; i<size; ++i)
	test_assert(equal(Z.get(r, i), mag(A.get(r, i))));
  }

  void diag()
  {
    length_type size = 256;

    Matrix<T>           Asup(rows, size + 16, T());
    Matrix<scalar_type> Zsup(rows, size + 16);

    typename Matrix<T>::subview_type A = Asup(Domain<2>(rows, size));
    typename Matrix<T>::subview_type Z = Zsup(Domain<2>(rows, size));

    std::cout << ovxx::assignment::diagnostics(Z, mag(A)) << std::endl;
  }
};

void defaults(Loop1P&) {}

int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vmagsq1<        float   >()); break;
  case  2: loop(t_vmagsq1<complex<float > >()); break;
  case  3: loop(t_vmagsq1<        double  >()); break;
  case  4: loop(t_vmagsq1<complex<double> >()); break;

  case 11: loop(t_vmag1<        float   >()); break;
  case 12: loop(t_vmag1<complex<float > >()); break;
  case 13: loop(t_vmag1<        double  >()); break;
  case 14: loop(t_vmag1<complex<double> >()); break;

  case 111: loop(t_vmag_dense_mat<float>()); break;
  case 112: loop(t_vmag_nondense_mat<float>()); break;

  case   0:
    std::cout
      << "vmagsq -- vector magnitude {squared}\n"
      << "    -1 -- vector element-wise magnitude squared --         float\n"
      << "    -2 -- vector element-wise magnitude squared -- complex<float >\n"
      << "    -3 -- vector element-wise magnitude squared --         double\n"
      << "    -4 -- vector element-wise magnitude squared -- complex<double>\n"
      << "   -11 -- vector element-wise magnitude         --         float\n"
      << "   -12 -- vector element-wise magnitude         -- complex<float >\n"
      << "   -13 -- vector element-wise magnitude         --         double\n"
      << "   -14 -- vector element-wise magnitude         -- complex<double>\n"
      << "  -111 --    dense matrix magnitude -- float\n"
      << "  -112 -- nondense matrix magnitude -- float\n"
      ;

  default:
    return 0;
  }
  return 1;
}
