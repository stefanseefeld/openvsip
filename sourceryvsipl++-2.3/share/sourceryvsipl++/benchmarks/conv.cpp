/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/conv.cpp
    @author  Jules Bergmann
    @date    2005-07-11
    @brief   VSIPL++ Library: Benchmark for 1D Convolution.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include <vsip/opt/dispatch_diagnostics.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;



/***********************************************************************
  Definitions
***********************************************************************/

template <support_region_type Supp,
	  typename            T>
struct t_conv1 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_conv1"; }
  float ops_per_point(length_type size)
  {
    length_type output_size;

    if      (Supp == support_full)
      output_size = ((size + coeff_size_ - 2)/Dec) + 1;
    else if (Supp == support_same)
      output_size = ((size-1)/Dec) + 1;
    else /* (Supp == support_min) */
      output_size = ((size-1)/Dec) - ((coeff_size_-1)/Dec) + 1;

    float ops = coeff_size_ * output_size *
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops / size;
  }

  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    length_type output_size;

    if      (Supp == support_full)
      output_size = ((size + coeff_size_ - 2)/Dec) + 1;
    else if (Supp == support_same)
      output_size = ((size-1)/Dec) + 1;
    else /* (Supp == support_min) */
      output_size = ((size-1)/Dec) - ((coeff_size_-1)/Dec) + 1;

    Vector<T>   in (size, T());
    Vector<T>   out(output_size);
    Vector<T>   coeff(coeff_size_, T());

    coeff(0) = T(1);
    coeff(1) = T(2);

    symmetry_type const       symmetry = nonsym;

    typedef Convolution<const_Vector, symmetry, Supp, T> conv_type;

    conv_type conv(coeff, Domain<1>(size), Dec);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      conv(in, out);
    t1.stop();
    
    time = t1.delta();
  }

  t_conv1(length_type coeff_size) : coeff_size_(coeff_size) {}

  void diag()
  {
    using namespace vsip_csl::dispatcher;
    typedef typename Dispatcher<op::conv<1, nonsym, Supp, T> >::backend backend;
    std::cout << "BE: " << Backend_name<backend>::name() << std::endl;
  }

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
  typedef complex<float> cf_type;

  switch (what)
  {
  case  1: loop(t_conv1<support_full, float>(loop.user_param_)); break;
  case  2: loop(t_conv1<support_same, float>(loop.user_param_)); break;
  case  3: loop(t_conv1<support_min,  float>(loop.user_param_)); break;

  case  4: loop(t_conv1<support_full, cf_type>(loop.user_param_)); break;
  case  5: loop(t_conv1<support_same, cf_type>(loop.user_param_)); break;
  case  6: loop(t_conv1<support_min,  cf_type>(loop.user_param_)); break;

  case 0:
    std::cout
      << "conv -- 1D convolution\n"
      << "   -1 -- float, support=full\n"
      << "   -2 -- float, support=same\n"
      << "   -3 -- float, support=min\n"
      << "   -4 -- complex<float>, support=full\n"
      << "   -5 -- complex<float>, support=same\n"
      << "   -6 -- complex<float>, support=min\n"
      << "\n"
      << " Parameters:\n"
      << "  -param N      -- size of coefficient vector (default 16)\n"
      << "  -start N      -- starting problem size 2^N (default 4 or 16 points)\n"
      << "  -loop_start N -- initial number of calibration loops (default 5000)\n"
      ;   

  default:
    return 0;
  }
  return 1;
}
