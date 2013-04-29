//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vthresh

#include <iostream>
#include <sal.h>

#include <vsip/random.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/core/ops_info.hpp>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace vsip;

template <typename T>
struct t_vthres_sal;

template <typename T>
struct t_vthr_sal;

template <>
struct t_vthres_sal<float> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vthres_sal"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 1*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    Vector<T, block_type>   result(size, T());
    T                       b = T(0.5);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const *p_A      = ext_A.ptr();
      T* p_result = ext_result.ptr();
      
      t1.start();
      marker1_start();
      for (size_t l=0; l<loop; ++l)
      {
	vthresx(const_cast<T*>(p_A),1, &b, p_result,1, size, 0);
      }
      marker1_stop();
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) >= b) ? A(i) : 0.f));
    }
    
    time = t1.delta();
  }
};



#if VSIP_IMPL_SAL_HAVE_VTHRX
template <>
struct t_vthr_sal<float> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vthr_sal"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 1*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    Vector<T, block_type>   result(size, T());
    T                       b = T(0.5);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const *p_A      = ext_A.ptr();
      T* p_result = ext_result.ptr();
      
      t1.start();
      marker1_start();
      for (size_t l=0; l<loop; ++l)
      {
	vthrx(const_cast<T*>(p_A),1, &b, p_result,1, size, 0);
      }
      marker1_stop();
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) >= b) ? A(i) : b));
    }
    
    time = t1.delta();
  }
};
#endif



template <typename T>
struct t_vthres_c : Benchmark_base
{
  char const *what() { return "t_vthres_c"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    T                       b = T(0.5);
    Vector<T, block_type>   result(size, T());

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const *p_A      = ext_A.ptr();
      T* p_result = ext_result.ptr();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	for (index_type i=0; i<size; ++i)
	  p_result[i] = (p_A[i] >= b) ? p_A[i] : 0.f;
      }
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) >= b) ? A(i) : 0.f));
    }
    
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
  case  1: loop(t_vthres_sal<float>()); break;
#if VSIP_IMPL_SAL_HAVE_VTHRX
  case  2: loop(t_vthr_sal<float>()); break;
#endif
  case 11: loop(t_vthres_c<float>()); break;
  case  0:
    std::cout
      << "SAL vthres\n"
      << "  -1 -- SAL vthresx (float) Z(i) = A(i) > b ? A(i) : 0\n"
#if VSIP_IMPL_SAL_HAVE_VTHRX
      << "  -2 -- SAL vthrx   (float) Z(i) = A(i) > b ? A(i) : b\n"
#else
      << " (-2 -- SAL vthrx function not available)\n"
#endif
      << " -11 -- C           (float) Z(i) = A(i) > b ? A(i) : 0\n"
      ;
  default:
    return 0;
  }
  return 1;
}
