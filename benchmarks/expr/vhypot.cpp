/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for vector hypot.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/diagnostics.hpp>

#include "../benchmarks.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise hypot
***********************************************************************/

template <typename T>
struct t_vhypot1 : Benchmark_base
{
  char const* what() { return "t_vhypot1"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = hypot(B,A);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), hypot(B.get(i),A.get(i))));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    vsip_csl::assign_diagnostics(C, hypot(B,A));
  }
};








void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vhypot1<float>()); break;

  case  0:
    std::cout
      << "vhypot -- vector hypot\n"
      << "                F  - float\n"
      << "   -1 -- vector element-wise hypot  -- F/F \n"
      ;
  default:
    return 0;
  }
  return 1;
}
