//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector atan/atan2.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "../benchmark.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise atan
***********************************************************************/

template <typename T>
struct t_vatan1 : Benchmark_base
{
  char const* what() { return "t_vatan1"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = atan(A);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), atan(A.get(i))));
  }

  void diag()
  {
    using ovxx::assignment::diagnostics;

    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    std::cout << diagnostics(C, atan(A)) << std::endl;
  }
};

/***********************************************************************
  Definitions - vector element-wise atan2
***********************************************************************/

template <typename T>
struct t_vatan21 : Benchmark_base
{
  char const* what() { return "t_vatan21"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = atan2(B,A);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), atan2(B.get(i),A.get(i))));
  }

  void diag()
  {
    using ovxx::assignment::diagnostics;

    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    std::cout << diagnostics(C, atan2(B,A)) << std::endl;
  }
};

void
defaults(Loop1P&)
{
}

int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vatan1<float>()); break;
  case 11: loop(t_vatan21<float>()); break;

  case  0:
    std::cout
      << "vatan -- vector atan/atan2\n"
      << "                F  - float\n"
      << "   -1 -- vector element-wise atan  -- F/F \n"
      << "  -11 -- vector element-wise atan2 -- F/F \n"
      ;
  default:
    return 0;
  }
  return 1;
}
