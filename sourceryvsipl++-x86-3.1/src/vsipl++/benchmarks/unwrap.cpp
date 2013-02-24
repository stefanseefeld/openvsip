/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/


/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/unwrap.hpp>
#include <vsip_csl/profile.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

#include "benchmarks.hpp"

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  VSIPL++ unwrap
***********************************************************************/

template <typename T>
struct t_unwrap : Benchmark_base
{
  char const* what() { return "t_unwrap_vector"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return -1; }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T> v_in(size, T());
    Vector<T> v_ref(size, T());
    Vector<T> v_out(size, T());
    
    v_ref = ramp(T(0.0), sign_ * T(2*M_PI/rate_), size);
    v_in = fmod((v_ref + M_PI), 2*M_PI) - M_PI;

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      vsip_csl::unwrap(v_out, v_in);
    t1.stop();

    time = t1.delta();
  }

  t_unwrap(length_type rate, int sign)
    : rate_(rate), sign_(sign) {}

  // member data.
  length_type rate_;
  int sign_;
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
  case   1: loop(t_unwrap<float>(4096,1)); break;
  case   2: loop(t_unwrap<float>(64,1)); break;
  case   3: loop(t_unwrap<float>(12,1)); break;
  case   4: loop(t_unwrap<float>(6,1)); break;
  case   5: loop(t_unwrap<float>(3,1)); break;

  case   0:
    std::cout
      << "unwrap -- remove jumps more than PI\n"
      << "   -1: vector, float, rate=4096\n"
      << "   -2: vector, float, rate=64\n"
      << "   -3: vector, float, rate=12\n"
      << "   -4: vector, float, rate=6\n"
      << "   -5: vector, float, rate=3\n"
      ;
  default: return 0;
  }
  return 1;
}
