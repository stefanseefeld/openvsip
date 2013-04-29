//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector sub

#include <iostream>

#include <vsip/initfin.hpp>

#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/assignment.hpp>
#include <vsip_csl/diagnostics.hpp>

#include "../benchmarks.hpp"

using namespace vsip;

/***********************************************************************
  Definitions - vector element-wise subtraction
***********************************************************************/

// Elementwise vector-subtraction, non-distributed (explicit Local_map)

template <typename T>
struct t_vsub1_nonglobal : Benchmark_base
{
  char const* what() { return "t_vsub1_nonglobal"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
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

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = A - B;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) - B.get(i)));
    
    time = t1.delta();
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
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case   1: loop(t_vsub1_nonglobal<float>()); break;
  case   2: loop(t_vsub1_nonglobal<complex<float> >()); break;

  case 0:
    std::cout
      << "vsub -- vector subtraction\n"
      << "single-precision:\n"
      << " Vector-Vector:\n"
      << "   -1 -- Vector<        float > - Vector<        float >\n"
      << "   -2 -- Vector<complex<float>> - Vector<complex<float>>\n"
      << "\n"
      ;

  default:
    return 0;
  }
  return 1;
}
