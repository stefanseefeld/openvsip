//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;

template <typename T>
struct t_hist : Benchmark_base
{
  char const* what() { return "t_histogram";}
  // 4 is worst-case scenario: 3 ops for computing the
  // bin slot, and 1 op for incrementing that.
  float ops_per_point(length_type) { return 4;}

  int riob_per_point(length_type) { return -1;}
  int wiob_per_point(length_type) { return -1;}
  int mem_per_point(length_type)  { return 2*sizeof(T);}

  void operator()(length_type size, length_type loop, float& time)
  {
    Rand<T> rgen(0);
    Vector<T> in = rgen.randu(size);
    Vector<int> hist(coeff_size_);
    Histogram<const_Vector, T> h(0, 1, coeff_size_);
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      hist = h(in);
    t1.stop();
    
    time = t1.delta();
  }

  t_hist(length_type coeff_size) : coeff_size_(coeff_size) {}

  length_type coeff_size_;
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.user_param_ = 16;
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_hist<float>(loop.user_param_)); break;
  case  2: loop(t_hist<int>(loop.user_param_)); break;
  case 0:
    std::cout
      << "histogram -- Histogram generation\n"
      << "   -1 -- float\n"
      << "   -2 -- int\n"
      << "\n"
      << " Parameters:\n"
      << "  -param N      -- size of histogram\n"
      << "  -start N      -- starting problem size 2^N (default 4 or 16 points)\n"
      << "  -loop_start N -- initial number of calibration loops (default 5000)\n"
      ;   

  default:
    return 0;
  }
  return 1;
}
