/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for SAL memory write.

#include <iostream>
#include <complex>

#include <vsip/random.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/core/ops_info.hpp>
#include <sal.h>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace vsip;

template <typename T, storage_format_type C = interleaved_complex>
struct t_memwrite_sal;

template <storage_format_type C>
struct t_memwrite_sal<float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_memwrite_sal"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T val = T(3);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T* pZ = ext_z.ptr();
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
	vfillx(&val, pZ, 1, size, 0);
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
      test_assert(Z.get(i) == val);
    
    time = t1.delta();
  }
};






void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float> cf_type;
  switch (what)
  {
  case  1: loop(t_memwrite_sal<float>()); break;
  case  0:
    std::cout
      << "memwrite -- SAL memory write\n"
      << "  -1: use vfillx to fill a vector with a scalar\n"
      ;
  default:
    return 0;
  }
  return 1;
}
