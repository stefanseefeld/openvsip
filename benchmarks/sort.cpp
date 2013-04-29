/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for sort.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/sort.hpp>
#include "benchmarks.hpp"

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions - vector element-wise divide
***********************************************************************/

template <typename T, bool ascend=true>
struct t_std_sort : Benchmark_base
{
  char const* what() { return "t_std_sort"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  typedef
    typename vsip::impl::conditional<
      ascend, std::less<T>, std::greater<T> >::type
    comp_type;

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());

    comp_type comp = comp_type();

    Rand<T> gen(0, 0);
    B = gen.randu(size);
    A = B;

    typedef vsip::Layout<1, vsip::row1_type, vsip::dense> layout_type;

    vsip::dda::Data<typename Vector<T>::block_type, vsip::dda::inout, layout_type> 
      raw(A.block());

    vsip_csl::profile::Timer t1;

    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      A = B;
      std::sort(raw.ptr(), raw.ptr() + raw.size(0), comp);
    }
    t1.stop();
    
    for (index_type i=0; i<(size-1); ++i)
      test_assert( !comp( A.get(i+1), A.get(i) ) );
    
    time = t1.delta();
  }

};

template <typename T, bool ascend=true>
struct t_sort_data_ip : Benchmark_base
{
  char const* what() { return "t_sort_data_ip"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  typedef
    typename vsip::impl::conditional<
      ascend, std::less<T>, std::greater<T> >::type
    comp_type;

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());

    comp_type comp = comp_type();

    Rand<T> gen(0, 0);
    B = gen.randu(size);
    A = B;

    vsip_csl::profile::Timer t1;

    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      A = B;
      sort_data(A, comp);
    }
    t1.stop();
    
    for (index_type i=0; i<(size-1); ++i)
      test_assert( !comp( A.get(i+1), A.get(i) ) );
    
    time = t1.delta();
  }

};

template <typename T, bool ascend=true>
struct t_sort_data_op : Benchmark_base
{
  char const* what() { return "t_sort_data_op"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  typedef
    typename vsip::impl::conditional<
      ascend, std::less<T>, std::greater<T> >::type
    comp_type;

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size, T());

    comp_type comp = comp_type();

    Rand<T> gen(0, 0);
    B = gen.randu(size);
    A = B;

    vsip_csl::profile::Timer t1;

    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      A = B;
      sort_data(A, C, comp);
    }
    t1.stop();
    
    for (index_type i=0; i<(size-1); ++i)
      test_assert( !comp( C.get(i+1), C.get(i) ) );
    
    time = t1.delta();
  }

};

template <typename T, bool ascend=true>
struct t_sort_indices : Benchmark_base
{
  char const* what() { return "t_sort_indices"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  typedef
    typename vsip::impl::conditional<
      ascend, std::less<T>, std::greater<T> >::type
    comp_type;

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>            B(size, T());
    Vector<index_type>   X(size);

    comp_type comp = comp_type();

    Rand<float> gen(0, 0);
    B = gen.randu(size) * 4096;

    vsip_csl::profile::Timer t1;

    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      sort_indices(X, B, comp);
    }
    t1.stop();
    
    for (index_type i=0; i<(size-1); ++i)
      test_assert( !comp( B.get(X.get(i+1)), B.get(X.get(i)) ) );
    
    time = t1.delta();
  }

};


void
defaults(Loop1P& loop)
{
  loop.fix_loop_ = true;
  loop.loop_start_ = 100;
  loop.stop_ = 18;
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_std_sort<float>()); break;
  case  2: loop(t_std_sort<int  >()); break;
  case  3: loop(t_std_sort<float, false>()); break;
  case  4: loop(t_std_sort<int  , false>()); break;

  case 11: loop(t_sort_data_ip<float>()); break;
  case 12: loop(t_sort_data_ip<int  >()); break;
  case 13: loop(t_sort_data_ip<float, false>()); break;
  case 14: loop(t_sort_data_ip<int  , false>()); break;

  case 21: loop(t_sort_data_op<float>()); break;
  case 22: loop(t_sort_data_op<int  >()); break;
  case 23: loop(t_sort_data_op<float, false>()); break;
  case 24: loop(t_sort_data_op<int  , false>()); break;

  case 31: loop(t_sort_indices<float>()); break;
  case 32: loop(t_sort_indices<int  >()); break;
  case 33: loop(t_sort_indices<float, false>()); break;
  case 34: loop(t_sort_indices<int  , false>()); break;

  default:
    printf("Sort benchmark\n");
    printf("   -1: float,  ascending, std::sort\n");
    printf("   -2:   int,  ascending, std::sort\n");
    printf("   -3: float, descending, std::sort\n");
    printf("   -4:   int, descending, std::sort\n");
    printf("  -11: float,  ascending, sort_data in-place\n");
    printf("  -12:   int,  ascending, sort_data in-place\n");
    printf("  -13: float, descending, sort_data in-place\n");
    printf("  -14:   int, descending, sort_data in-place\n");
    printf("  -21: float,  ascending, sort_data out-of-place\n");
    printf("  -22:   int,  ascending, sort_data out-of-place\n");
    printf("  -23: float, descending, sort_data out-of-place\n");
    printf("  -24:   int, descending, sort_data out-of-place\n");
    printf("  -31: float,  ascending, sort_indices\n");
    printf("  -32:   int,  ascending, sort_indices\n");
    printf("  -33: float, descending, sort_indices\n");
    printf("  -34:   int, descending, sort_indices\n");
    return 0;
  }
  return 1;
}
