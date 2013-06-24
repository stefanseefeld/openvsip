//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for Correlation.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include "benchmark.hpp"

using namespace ovxx;

template <support_region_type Supp,
	  typename            T>
struct t_corr1 : Benchmark_base
{
  char const* what() { return "t_corr1"; }
  float ops_per_point(length_type size)
  {
    length_type output_size = this->my_output_size(size);
    float ops = ref_size_ * output_size *
       (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops / size;
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type) { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    length_type output_size = this->my_output_size(size);

    Vector<T>   in (size, T());
    Vector<T>   out(output_size);
    Vector<T>   ref(ref_size_, T());

    ref(0) = T(1);
    ref(1) = T(2);

    typedef Correlation<const_Vector, Supp, T> corr_type;

    corr_type corr((Domain<1>(ref_size_)), Domain<1>(size));

    timer t1;
    for (index_type l=0; l<loop; ++l)
     corr(bias_, ref, in, out);
    time = t1.elapsed();
  }

  t_corr1(length_type ref_size, bias_type bias)
    : ref_size_(ref_size),
      bias_    (bias)
  {}

  
  length_type my_output_size(length_type size)
  {
    if      (Supp == support_full)
      return size + ref_size_ - 1;
    else if (Supp == support_same)
      return size;
    else /* (Supp == support_min) */
      return size - ref_size_ + 1;
  }
  

  length_type ref_size_;
  bias_type   bias_;
};



// Benchmark performance of a Correlation_impl object
// (requires ImplTag to select implementation)

template <typename            ImplTag,
	  support_region_type Supp,
	  typename            T>
struct t_corr2 : Benchmark_base
{
  char const* what() { return "t_corr2"; }
  float ops_per_point(length_type size)
  {
    length_type output_size = this->my_output_size(size);
    float ops = ref_size_ * output_size *
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops / size;
  }

  int riob_per_point(length_type) { return -1*(int)sizeof(T); }
  int wiob_per_point(length_type) { return -1*(int)sizeof(T); }
  int mem_per_point(length_type) { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    using namespace dispatcher;

    length_type output_size = this->my_output_size(size);

    Vector<T>   in (size, T());
    Vector<T>   out(output_size);
    Vector<T>   ref(ref_size_, T());

    ref(0) = T(1);
    ref(1) = T(2);

    typedef typename 
      Evaluator<op::corr<1, Supp, T, 0, alg_time>, ImplTag>::backend_type
      corr_type;

    corr_type corr((Domain<1>(ref_size_)), Domain<1>(size));

    timer t1;
    
    for (index_type l=0; l<loop; ++l)
     corr.correlate(bias_, ref, in, out);
    time = t1.elapsed();
  }

  t_corr2(length_type ref_size, bias_type bias)
    : ref_size_(ref_size),
      bias_    (bias)
  {}

  
  length_type my_output_size(length_type size)
  {
    if      (Supp == support_full)
      return size + ref_size_ - 1;
    else if (Supp == support_same)
      return size;
    else /* (Supp == support_min) */
      return size - ref_size_ + 1;
  }
  

  length_type ref_size_;
  bias_type   bias_;
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.user_param_ = 16;
}



int
benchmark(Loop1P& loop, int what)
{
  length_type M = loop.user_param_;
  using namespace dispatcher;

  typedef float T;
  typedef complex<float> CT;

  switch (what)
  {
  case  1: loop(t_corr1<support_full, T>(M, biased)); break;
  case  2: loop(t_corr1<support_same, T>(M, biased)); break;
  case  3: loop(t_corr1<support_min,  T>(M, biased)); break;

  case  4: loop(t_corr1<support_full, T>(M, unbiased)); break;
  case  5: loop(t_corr1<support_same, T>(M, unbiased)); break;
  case  6: loop(t_corr1<support_min,  T>(M, unbiased)); break;

  case  7: loop(t_corr1<support_full, CT >(M, biased)); break;
  case  8: loop(t_corr1<support_same, CT >(M, biased)); break;
  case  9: loop(t_corr1<support_min,  CT >(M, biased)); break;

  case  10: loop(t_corr1<support_full, CT >(M, unbiased)); break;
  case  11: loop(t_corr1<support_same, CT >(M, unbiased)); break;
  case  12: loop(t_corr1<support_min,  CT >(M, unbiased)); break;

  case  25: loop(t_corr2<be::generic, support_full, T>(M, biased)); break;
  case  26: loop(t_corr2<be::generic, support_same, T>(M, biased)); break;
  case  27: loop(t_corr2<be::generic, support_min,  T>(M, biased)); break;

  case  28: loop(t_corr2<be::generic, support_full, T>(M, unbiased)); break;
  case  29: loop(t_corr2<be::generic, support_same, T>(M, unbiased)); break;
  case  30: loop(t_corr2<be::generic, support_min,  T>(M, unbiased)); break;

  case  31: loop(t_corr2<be::generic, support_full, CT >(M, biased)); break;
  case  32: loop(t_corr2<be::generic, support_same, CT >(M, biased)); break;
  case  33: loop(t_corr2<be::generic, support_min,  CT >(M, biased)); break;

  case  34: loop(t_corr2<be::generic, support_full, CT >(M, unbiased)); break;
  case  35: loop(t_corr2<be::generic, support_same, CT >(M, unbiased)); break;
  case  36: loop(t_corr2<be::generic, support_min,  CT >(M, unbiased)); break;

  case 0:
    std::cout
      << "corr -- correlation\n"
      << "  option dim backend support           type biased\n"
      << "  ------ --- ------- ------- -------------- ------\n"
      << "     -1   1    n/a     full           float   yes\n"
      << "     -2   1    n/a     same           float   yes\n"
      << "     -3   1    n/a     min            float   yes\n"
      << "     -4   1    n/a     full           float    no\n"
      << "     -5   1    n/a     same           float    no\n"
      << "     -6   1    n/a     min            float    no\n"
      << "     -7   1    n/a     full  complex<float>   yes\n"
      << "     -8   1    n/a     same  complex<float>   yes\n"
      << "     -9   1    n/a     min   complex<float>   yes\n"
      << "    -10   1    n/a     full  complex<float>    no\n"
      << "    -11   1    n/a     same  complex<float>    no\n"
      << "    -12   1    n/a     min   complex<float>    no\n"
      << "    -13   2    [O]     full           float   yes\n"
      << "    -14   2    [O]     same           float   yes\n"
      << "    -15   2    [O]     min            float   yes\n"
      << "    -16   2    [O]     full           float    no\n"
      << "    -17   2    [O]     same           float    no\n"
      << "    -18   2    [O]     min            float    no\n"
      << "    -19   2    [O]     full  complex<float>   yes\n"
      << "    -20   2    [O]     same  complex<float>   yes\n"
      << "    -21   2    [O]     min   complex<float>   yes\n"
      << "    -22   2    [O]     full  complex<float>    no\n"
      << "    -23   2    [O]     same  complex<float>    no\n"
      << "    -24   2    [O]     min   complex<float>    no\n"
      << "    -25   2    [G]     full           float   yes\n"
      << "    -26   2    [G]     same           float   yes\n"
      << "    -27   2    [G]     min            float   yes\n"
      << "    -28   2    [G]     full           float    no\n"
      << "    -29   2    [G]     same           float    no\n"
      << "    -30   2    [G]     min            float    no\n"
      << "    -31   2    [G]     full  complex<float>   yes\n"
      << "    -32   2    [G]     same  complex<float>   yes\n"
      << "    -33   2    [G]     min   complex<float>   yes\n"
      << "    -34   2    [G]     full  complex<float>    no\n"
      << "    -35   2    [G]     same  complex<float>    no\n"
      << "    -36   2    [G]     min   complex<float>    no\n"
      << "  \n"
      << "   Notes:\n"
      << "    [O] -- optimized generic backend\n"
      << "    [G] -- generic backend\n"
      << "  \n"
      << "   Parameters:\n"
      << "    -param N      -- size of coefficient vector\n"
      << "                     (default 16)\n"
      << "    -start N      -- starting problem size 2^N\n"
      << "                     (default 4 or 16 points)\n"
      << "    -loop_start N -- initial number of calibration\n"
      << "                     loops (default 5000)\n"
      ;

  default:
    return 0;
  }
  return 1;
}
