/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/vmul.cpp
    @author  Jules Bergmann
    @date    2006-01-23
    @brief   VSIPL++ Library: Benchmark for vector multiply using C arrays.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/core/profile.hpp>

#include <vsip_csl/test.hpp>
#include "../loop.hpp"
#include "../benchmarks.hpp"

using namespace vsip;

using impl::Stride_unit_dense;
using impl::Cmplx_inter_fmt;



/***********************************************************************
  Definitions -- generic vector-multiply
***********************************************************************/


template <typename T>
inline void
vmul(
   int	n,
   T*	A,
   T*	B,
   T*	R)
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
svmul(
   int	n,
   T	alpha,
   T*	B,
   T*	R)
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
    typedef impl::Layout<1, row1_type, Stride_unit_dense, Cmplx_inter_fmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    A(0) = T(3);
    B(0) = T(4);

    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<block_type> ext_a(A.block(), impl::SYNC_IN);
      impl::Ext_data<block_type> ext_b(B.block(), impl::SYNC_IN);
      impl::Ext_data<block_type> ext_c(C.block(), impl::SYNC_OUT);
      
      T* pa = ext_a.data();
      T* pb = ext_b.data();
      T* pc = ext_c.data();
    
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
    typedef impl::Layout<1, row1_type, Stride_unit_dense, Cmplx_inter_fmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    A(0) = T(3);
    B(0) = T(4);

    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<block_type> ext_a(A.block(), impl::SYNC_IN);
      impl::Ext_data<block_type> ext_b(B.block(), impl::SYNC_IN);
      impl::Ext_data<block_type> ext_c(C.block(), impl::SYNC_OUT);
      
      T* pa = ext_a.data();
      T* pb = ext_b.data();
      T* pc = ext_c.data();
    
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
    typedef impl::Layout<1, row1_type, Stride_unit_dense, Cmplx_inter_fmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   C(size);

    T alpha = T(3);

    A(0) = T(4);
    
    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<block_type> ext_a(A.block(), impl::SYNC_IN);
      impl::Ext_data<block_type> ext_c(C.block(), impl::SYNC_OUT);
      
      T* pa = ext_a.data();
      T* pc = ext_c.data();
    
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
