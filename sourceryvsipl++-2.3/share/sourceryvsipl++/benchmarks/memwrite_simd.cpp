/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/memwrite_simd.cpp
    @author  Jules Bergmann
    @date    2006-10-13
    @brief   VSIPL++ Library: Benchmark for SIMD memory write.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <complex>

#include <vsip/random.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/opt/extdata.hpp>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace std;
using namespace vsip;

using impl::Stride_unit_dense;
using impl::Cmplx_inter_fmt;
using impl::Cmplx_split_fmt;



/***********************************************************************
  Definitions - vector element-wise multiply
***********************************************************************/

template <typename T,
	  typename ComplexFmt = Cmplx_inter_fmt>
struct t_memwrite_simd : Benchmark_base
{
  char const* what() { return "t_memwrite_simd"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef vsip::impl::simd::Simd_traits<T> S;
    typedef typename S::simd_type            simd_type;

    typedef impl::Layout<1, row1_type, Stride_unit_dense, Cmplx_inter_fmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T val = T(3);

    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<block_type> ext_z(Z.block(), impl::SYNC_OUT);
    
      T* pZ = ext_z.data();
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	T* ptr = pZ;
	simd_type reg0 = S::load_scalar_all(val);
	for (index_type i=0; i<size; i+=S::vec_size)
        {
	  S::store(ptr, reg0);
	  ptr += S::vec_size;
	}
      }
      t1.stop();
    }
    
    for (index_type i=0; i<size; ++i)
      test_assert(Z.get(i) == val);
    
    time = t1.delta();
  }
};



template <typename T,
	  typename ComplexFmt = Cmplx_inter_fmt>
struct t_memwrite_simd_r4 : Benchmark_base
{
  char const* what() { return "t_memwrite_simd_r4"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef vsip::impl::simd::Simd_traits<T> S;
    typedef typename S::simd_type            simd_type;

    typedef impl::Layout<1, row1_type, Stride_unit_dense, Cmplx_inter_fmt> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T val = T(3);

    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<block_type> ext_z(Z.block(), impl::SYNC_OUT);
    
      T* pZ = ext_z.data();
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	T* ptr = pZ;
	simd_type reg0 = S::load_scalar_all(val);

	length_type n = size;

	while (n >= 4*S::vec_size)
	{
	  S::store(ptr + 0*S::vec_size, reg0);
	  S::store(ptr + 1*S::vec_size, reg0);
	  S::store(ptr + 2*S::vec_size, reg0);
	  S::store(ptr + 3*S::vec_size, reg0);
	  ptr += 4*S::vec_size;
	  n   -= 4*S::vec_size;
	}

	while (n >= S::vec_size)
	{
	  S::store(ptr + 0*S::vec_size, reg0);
	  ptr += 1*S::vec_size;
	  n   -= 1*S::vec_size;
	}
      }
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
  case  1: loop(t_memwrite_simd<float>()); break;
  case  2: loop(t_memwrite_simd_r4<float>()); break;

  case  0:
    std::cout
      << "memwrite -- SIMD memory write bandwidth\n"
      << "  -1 -- write a float scalar into all elements of a view\n"
      << "        using an explicit SIMD loop\n"
      << "  -2 -- same using a loop unrolled 4 times\n"
      ;
  default: return 0;
  }
  return 1;
}
