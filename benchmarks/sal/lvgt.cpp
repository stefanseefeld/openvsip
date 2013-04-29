//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for lvgt

#include <iostream>
#include <sal.h>

#include <vsip/random.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/core/ops_info.hpp>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace vsip;

/***********************************************************************
  Definitions - vector element-wise multiply
***********************************************************************/

template <typename T>
struct t_lvgt_sal;

template <>
struct t_lvgt_sal<float> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_lvgt_sal"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    Vector<T, block_type>   B     (size, T());
    Vector<T, block_type>   result(size, T());

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::in> ext_B(B.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const* p_A = ext_A.ptr();
      T const* p_B = ext_B.ptr();
      T* p_result = ext_result.ptr();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	lvgtx(const_cast<T*>(p_A),1, const_cast<T*>(p_B),1, p_result,1, size, 0);
      }
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) > B(i)) ? 1.f : 0.f));
    }
    
    time = t1.delta();
  }
};



template <typename T>
struct t_threshold_sal;

template <>
struct t_threshold_sal<float> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_threshold_sal"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    Vector<T, block_type>   B     (size, T());
    Vector<T, block_type>   result(size, T());

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::in> ext_B(B.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const* p_A = ext_A.ptr();
      T const* p_B = ext_B.ptr();
      T* p_result = ext_result.ptr();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	lvgtx(const_cast<T*>(p_A),1, const_cast<T*>(p_B),1, p_result,1, size, 0);
	vmulx(const_cast<T*>(p_A),1, p_result,1, p_result,1, size, 0);
      }
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) > B(i)) ? A(i) : 0.f));
    }
    
    time = t1.delta();
  }
};



template <typename T>
struct t_lvgt_c : Benchmark_base
{
  char const *what() { return "t_lvgt_c"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    Vector<T, block_type>   B     (size, T());
    Vector<T, block_type>   result(size, T());

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::in> ext_B(B.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const *p_A = ext_A.ptr();
      T const *p_B = ext_B.ptr();
      T *p_result = ext_result.ptr();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	for (index_type i=0; i<size; ++i)
	  p_result[i] = (p_A[i] > p_B[i]) ? 1.f : 0.f;
      }
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) > B(i)) ? 1.f : 0.f));
    }
    
    time = t1.delta();
  }
};



template <typename T>
struct t_threshold_c : Benchmark_base
{
  char const *what() { return "t_threshold_c"; }
  int ops_per_point(size_t)  { return 1; }

  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A     (size, T());
    Vector<T, block_type>   B     (size, T());
    Vector<T, block_type>   result(size, T());

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    vsip_csl::profile::Timer t1;

    { // Control dda::Data scope.
      dda::Data<block_type, dda::in> ext_A(A.block());
      dda::Data<block_type, dda::in> ext_B(B.block());
      dda::Data<block_type, dda::out> ext_result(result.block());
    
      T const *p_A = ext_A.ptr();
      T const *p_B = ext_B.ptr();
      T *p_result = ext_result.ptr();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	for (index_type i=0; i<size; ++i)
	  p_result[i] = (p_A[i] > p_B[i]) ? p_A[i] : 0.f;
      }
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      test_assert(equal<T>(result.get(i), (A(i) > B(i)) ? A(i) : 0.f));
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
  case  1: loop(t_lvgt_sal<float>()); break;
  case  2: loop(t_threshold_sal<float>()); break;
  case 11: loop(t_lvgt_c<float>()); break;
  case 12: loop(t_threshold_c<float>()); break;
  case  0:
    std::cout
      << "SAL lvgt\n"
      << "  -1 -- SAL lvgtx      (float) Z(i) = A(i) > B(i) ? 1    : 0\n"
      << "  -2 -- SAL lvgtx/vmul (float) Z(i) = A(i) > B(i) ? A(i) : 0\n"
      << " -11 -- C (float) Z(i) = A(i) > B(i) ? 1    : 0\n"
      << " -12 -- C (float) Z(i) = A(i) > B(i) ? A(i) : 0\n"
      ;
  default:
    return 1;
  }
  return 1;
}
