//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for matrix-matrix products.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include "benchmark.hpp"

using namespace vsip;

// Matrix-matrix product benchmark class.

template <typename T>
struct t_prod1 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_prod1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    length_type N = M;
    length_type P = M;

    Matrix<T>   A (M, N, T(1));
    Matrix<T>   B (N, P, T(1));
    Matrix<T>   Z (M, P, T(1));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = prod(A, B);
    time = t1.elapsed();
  }

  t_prod1() {}
};



// Matrix-matrix product (with hermetian) benchmark class.

template <typename T>
struct t_prodh1 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_prodh1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    length_type N = M;
    length_type P = M;

    Matrix<T>   A (M, N, T());
    Matrix<T>   B (P, N, T());
    Matrix<T>   Z (M, P, T());

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = prodh(A, B);
    time = t1.elapsed();
  }

  t_prodh1() {}
};



// Matrix-matrix product (with tranpose) benchmark class.

template <typename T>
struct t_prodt1 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_prodt1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    length_type N = M;
    length_type P = M;

    Matrix<T>   A (M, N, T());
    Matrix<T>   B (P, N, T());
    Matrix<T>   Z (M, P, T());

    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = prodt(A, B);
    time = t1.elapsed();
  }

  t_prodt1() {}
};



// Matrix-matrix product benchmark class with particular ImplTag.

template <typename ImplTag,
	  typename T>
struct t_prod2 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_prod2"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    using namespace ovxx::dispatcher;

    typedef Dense<2, T, row2_type> a_block_type;
    typedef Dense<2, T, row2_type> b_block_type;
    typedef Dense<2, T, row2_type> z_block_type;

    typedef Evaluator<op::prod, ImplTag,
      void(z_block_type &, a_block_type const &, b_block_type const &)>
      evaluator_type;

    length_type N = M;
    length_type P = M;

    Matrix<T, a_block_type>   A(M, N, T());
    Matrix<T, b_block_type>   B(N, P, T());
    Matrix<T, z_block_type>   Z(M, P, T());

    timer t1;
    for (index_type l=0; l<loop; ++l)
      evaluator_type::exec(Z.block(), A.block(), B.block());
    time = t1.elapsed();
  }

  t_prod2() {}
};

void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.stop_  = 8;
}



int
benchmark(Loop1P& loop, int what)
{
  using namespace ovxx::dispatcher;

  switch (what)
  {
  case  1: loop(t_prod1<float>()); break;
  case  2: loop(t_prod1<complex<float> >()); break;

  case  3: loop(t_prod2<be::generic, float>()); break;
  case  4: loop(t_prod2<be::generic, complex<float> >()); break;

#if VSIP_IMPL_HAVE_BLAS
  case  5: loop(t_prod2<be::blas, float>()); break;
    // The BLAS backend doesn't handle split-complex, so don't attempt
    // to instantiate that code if we are using split-complex blocks.
# if !OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT
  case  6: loop(t_prod2<be::blas, complex<float> >()); break;
# endif
#endif

  case  11: loop(t_prodt1<float>()); break;
  case  12: loop(t_prodt1<complex<float> >()); break;
  case  13: loop(t_prodh1<complex<float> >()); break;

  case    0:
    std::cout
      << "prod -- matrix-matrix product\n"
      << "    -1 -- default implementation, float\n"
      << "    -2 -- default implementation, complex<float>\n"
      << "    -3 -- generic implementation, float\n"
      << "    -4 -- generic implementation, complex<float>\n"
      << "    -5 --    BLAS implementation, float\n"
      << "    -6 --    BLAS implementation, complex<float> {interleaved only}\n"
      << "   -11 -- default impl with transpose, float\n"
      << "   -12 -- default impl with transpose, complex<float>\n"
      << "   -13 -- default impl with hermetian, complex<float>\n"
      ;
  default: return 0;
  }
  return 1;
}
