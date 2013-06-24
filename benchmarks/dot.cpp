//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for dot-produtcs.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <ovxx/dispatch.hpp>
#include "benchmark.hpp"

using namespace vsip;
using namespace ovxx::dispatcher;

// Dot-product benchmark class.

template <typename T>
struct t_dot1 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_dot1"; }
  float ops_per_point(length_type)
  {
    float ops = (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type) { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A (size, T());
    Vector<T>   B (size, T());
    T r = T();

    A(0) = T(3);
    B(0) = T(4);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      r = dot(A, B);
    time = t1.elapsed();

    if (r != T(3)*T(4))
      abort();
  }

  t_dot1() {}
};



// Dot-product benchmark class with particular ImplTag.

template <typename ImplTag,
          typename T,
          bool     IsValid = Evaluator<
            op::dot, ImplTag,
            T(typename Vector<T>::block_type const&, 
              typename Vector<T>::block_type const&) >::ct_valid>
struct t_dot2 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_dot2"; }
  float ops_per_point(length_type)
  {
    float ops = (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type) { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A (size, T());
    Vector<T>   B (size, T());
    T r = T();

    A(0) = T(3);
    B(0) = T(4);

    typedef typename Vector<T>::block_type block_type;

    typedef Evaluator<op::dot, 
      ImplTag, T(block_type const&, block_type const&) > 
      Eval;

    assert(Eval::ct_valid);
  
    timer t1;
    for (index_type l=0; l<loop; ++l)
      r = Eval::exec(A.block(), B.block());
    time = t1.elapsed();

    test_assert(r == T(3)*T(4));
  }

  t_dot2() {}
};


template <typename ImplTag,
              typename T>
struct t_dot2<ImplTag, T, false> : Benchmark_base
{
  void operator()(length_type, length_type, float& time)
  {
    std::cout << "t_dot2: evaluator not implemented\n";
    time = 0;
    abort();
  }
}; 

void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
}



int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_dot1<float>()); break;
  case  2: loop(t_dot1<complex<float> >()); break;

  case  3: loop(t_dot2<be::generic, float>()); break;
  case  4: loop(t_dot2<be::generic, complex<float> >()); break;

#if VSIP_IMPL_HAVE_BLAS
  case  5: loop(t_dot2<be::blas, float>()); break;
  case  6: loop(t_dot2<be::blas, complex<float> >()); break;
#endif

#if VSIP_IMPL_HAVE_CUDA
  case  7: loop(t_dot2<be::cuda, float>()); break;
  case  8: loop(t_dot2<be::cuda, complex<float> >()); break;
#endif

  case  0:
    std::cout
      << "dot -- dot product\n"
      << "  -1 -- float\n"
      << "  -2 -- complex<float>\n"
      << "  -3 -- float          -- generic backend\n"
      << "  -4 -- complex<float> -- generic backend\n"
      << "  -5 -- float          -- BLAS backend\n"
      << "  -6 -- complex<float> -- BLAS backend\n"
      << "  -7 -- float          -- CUDA backend\n"
      << "  -8 -- complex<float> -- CUDA backend\n"
      << "\n"
      << " Parameters:\n"
      << "  -start N      -- starting problem size 2^N (default 4 or 16 points)\n"
      << "  -loop_start N -- initial number of calibration loops (default 5000)\n"
      ;

  default: return 0;
  }
  return 1;
}
