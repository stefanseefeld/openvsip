//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector add

#include <iostream>

#include <vsip/initfin.hpp>

#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include "../benchmark.hpp"

using namespace vsip;

/***********************************************************************
  Definitions - vector element-wise addition
***********************************************************************/

// Elementwise vector-addition, non-distributed (explicit Local_map)

template <typename T>
struct t_vadd1_nonglobal : Benchmark_base
{
  char const* what() { return "t_vadd1_nonglobal"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::add; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    typedef Dense<1, T, row1_type, Local_map> block_type;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = A + B;
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) + B.get(i)));
  }
};


/***********************************************************************
  Definitions
***********************************************************************/

void
defaults(Loop1P&)
{
}



int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case   1: loop(t_vadd1_nonglobal<float>()); break;
  case   2: loop(t_vadd1_nonglobal<complex<float> >()); break;

  case 0:
    std::cout
      << "vadd -- vector addition\n"
      << "single-precision:\n"
      << " Vector-Vector:\n"
      << "   -1 -- Vector<        float > + Vector<        float >\n"
      << "   -2 -- Vector<complex<float>> + Vector<complex<float>>\n"
      << "\n"
      ;

  default:
    return 0;
  }
  return 1;
}
