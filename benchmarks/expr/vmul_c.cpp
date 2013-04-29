//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector multiply using C arrays.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/profile.hpp>

#include <vsip_csl/test.hpp>
#include "../loop.hpp"
#include "../benchmarks.hpp"

using namespace vsip;

/***********************************************************************
  Definitions -- generic vector-multiply
***********************************************************************/


template <typename T>
inline void
vmul(int n, T const *A, T const *B, T *R)
{
  while (n)
  {
    *R = *A * *B;
    R++; A++; B++;
    n--;
  }
}



template <typename T>
inline void
svmul(int n, T alpha, T const *B, T *R)
{
  while (n)
  {
    *R = alpha * *B;
    R++; B++;
    n--;
  }
}



/***********************************************************************
  t_vmul1 -- Benchmark view-view vector multiply
***********************************************************************/

template <typename T>
struct t_vmul1 : Benchmark_base
{
  char const* what() { return "t_vmul1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    A(0) = T(3);
    B(0) = T(4);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
      
      T const * pa = ext_a.ptr();
      T const * pb = ext_b.ptr();
      T* pc = ext_c.ptr();
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
	for (index_type j=0; j<size; ++j)
	  pc[j] = pa[j] * pb[j];
      }
      t1.stop();
    }
      
    if (!equal(C(0), T(12)))
    {
      std::cout << "t_vmul1: ERROR" << std::endl
		<< "       : C(0) = " << C(0) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



/***********************************************************************
  t_vmul2 -- Benchmark view-view vector multiply
***********************************************************************/

template <typename T>
struct t_vmul2 : Benchmark_base
{
  char const* what() { return "t_vmul2"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    A(0) = T(3);
    B(0) = T(4);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
      
      T const * pa = ext_a.ptr();
      T const * pb = ext_b.ptr();
      T* pc = ext_c.ptr();
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	vmul(size, pa, pb, pc);
      t1.stop();
    }
      
    if (!equal(C(0), T(12)))
    {
      std::cout << "t_vmul2: ERROR" << std::endl
		<< "       : C(0) = " << C(0) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



/***********************************************************************
  t_svmul1 -- Benchmark scalar-view vector multiply
***********************************************************************/

template <typename T>
struct t_svmul1 : Benchmark_base
{
  char const* what() { return "t_svmul1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   C(size);

    T alpha = T(3);

    A(0) = T(4);
    
    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
      
      T const * pa = ext_a.ptr();
      T* pc = ext_c.ptr();
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	svmul(size, alpha, pa, pc);
      t1.stop();
    }
    
    test_assert(equal(C(0), T(12)));
    
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
  switch (what)
  {
  case  1: loop(t_vmul1<float>()); break;
  case  2: loop(t_vmul1<complex<float> >()); break;

  case  6: loop(t_vmul2<float>()); break;
  case  7: loop(t_vmul2<complex<float> >()); break;

  case 11: loop(t_svmul1<float>()); break;
  case 12: loop(t_svmul1<complex<float> >()); break;

  case  0:
    std::cout
      << "vmul_c -- vector multiply using C arrays\n"
      << "     -1 -- view-view elementwise multiply -- float\n"
      << "     -2 -- view-view elementwise multiply -- complex<float>\n"
      << "     -6 -- view-view vector multiply      -- float\n"
      << "     -7 -- view-view vector multiply      -- complex<float>\n"
      << "    -11 -- scalar-view vector multiply    -- float\n"
      << "    -12 -- scalar-view vector multiply    -- complex<float>\n"
      ;
  default:
    return 0;
  }
  return 1;
}
