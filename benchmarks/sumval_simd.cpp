/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for SIMD sumval.

#include <iostream>
#include <complex>

#include <vsip/random.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/dda.hpp>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace std;
using namespace vsip;

/***********************************************************************
  SIMD sumval (overhead included)
***********************************************************************/

template <typename T,
	  storage_format_type C = interleaved_complex>
struct t_sumval_simd : Benchmark_base
{
  typedef vsip::impl::simd::Simd_traits<T> S;
  typedef typename S::simd_type            simd_type;

  char const* what() { return "t_sumval_simd"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void use(simd_type) {}

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T res = T(0);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T* pZ = ext_z.ptr();
      T psum[S::vec_size];

      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	T* ptr = pZ;
	simd_type vsum = S::zero();
	simd_type reg0;
	for (index_type i=0; i<size; i+=S::vec_size)
        {
	  reg0 = S::load(ptr);
	  vsum = S::add(vsum, reg0);
	  ptr += S::vec_size;
	}
	S::store(psum, vsum);
	res = psum[0];
	for (index_type i=1; i<S::vec_size; i+=1)
	  res += psum[i];
      }
      
      t1.stop();
    }

    test_assert(res == T(size*2));

    
    time = t1.delta();
  }
};



/***********************************************************************
  SIMD sumval (overhead not included)
***********************************************************************/

template <typename T,
	  storage_format_type C = interleaved_complex>
struct t_sumval_simd_no : Benchmark_base // no-overheads
{
  typedef vsip::impl::simd::Simd_traits<T> S;
  typedef typename S::simd_type            simd_type;

  char const* what() { return "t_sumval_simd"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void use(simd_type) {}

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T res = T(0);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T* pZ = ext_z.ptr();
      T psum[S::vec_size];

      simd_type vsum = S::zero();
      simd_type reg0;

      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	T* ptr = pZ;
	vsum = S::zero();
	for (index_type i=0; i<size; i+=S::vec_size)
        {
	  reg0 = S::load(ptr);
	  vsum = S::add(vsum, reg0);
	  ptr += S::vec_size;
	}
      }
      t1.stop();

      // These overheads aren't included in timing.
      S::store(psum, vsum);
      res = psum[0];
      for (index_type i=1; i<S::vec_size; i+=1)
	res += psum[i];
    }

    test_assert(res == T(size*2));
    
    time = t1.delta();
  }
};



/***********************************************************************
  SIMD sumval (overhead included, loop unrolled 4 times)
***********************************************************************/

template <typename T,
	  storage_format_type C = interleaved_complex>
struct t_sumval_simd_r4 : Benchmark_base
{
  char const* what() { return "t_sumval_simd_r4"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef vsip::impl::simd::Simd_traits<T> S;
    typedef typename S::simd_type            simd_type;

    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T res = T(0);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T* pZ = ext_z.ptr();
      T psum[S::vec_size];
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	T* ptr = pZ;
	simd_type reg0;
	simd_type reg1;
	simd_type reg2;
	simd_type reg3;
	simd_type vsum0 = S::zero();
	simd_type vsum1 = S::zero();
	simd_type vsum2 = S::zero();
	simd_type vsum3 = S::zero();

	length_type n = size;

	while (n >= 4*S::vec_size)
	{
	  reg0 = S::load(ptr + 0*S::vec_size);
	  reg1 = S::load(ptr + 1*S::vec_size);
	  reg2 = S::load(ptr + 2*S::vec_size);
	  reg3 = S::load(ptr + 3*S::vec_size);
	  vsum0 = S::add(reg0, vsum0);
	  vsum1 = S::add(reg1, vsum1);
	  vsum2 = S::add(reg2, vsum2);
	  vsum3 = S::add(reg3, vsum3);
	  ptr += 4*S::vec_size;
	  n   -= 4*S::vec_size;
	}

	while (n >= S::vec_size)
	{
	  reg0 = S::load(ptr + 0*S::vec_size);
	  vsum0 = S::add(reg0, vsum0);
	  ptr += 1*S::vec_size;
	  n   -= 1*S::vec_size;
	}

	vsum0 = S::add(vsum0, vsum1);
	vsum2 = S::add(vsum2, vsum3);
	vsum0 = S::add(vsum0, vsum2);

	S::store(psum, vsum0);
	res = psum[0];
	for (index_type i=1; i<S::vec_size; i+=1)
	  res += psum[i];
      }
      t1.stop();
    }
    
    test_assert(res == T(size*2));
    
    time = t1.delta();
  }
};



/***********************************************************************
  SIMD sumval (overhead not included, loop unrolled 4 times)
***********************************************************************/

template <typename T,
	  storage_format_type C = interleaved_complex>
struct t_sumval_simd_r4_no : Benchmark_base
{
  char const* what() { return "t_sumval_simd_r4_no"; }
  int ops_per_point(size_t)  { return 1; }
  int riob_per_point(size_t) { return 1*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef vsip::impl::simd::Simd_traits<T> S;
    typedef typename S::simd_type            simd_type;

    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   Z(size, T(2));
    T res = T(0);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T* pZ = ext_z.ptr();
      T psum[S::vec_size];

      simd_type reg0;
      simd_type reg1;
      simd_type reg2;
      simd_type reg3;
      simd_type vsum0 = S::zero();
      simd_type vsum1 = S::zero();
      simd_type vsum2 = S::zero();
      simd_type vsum3 = S::zero();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
      {
	T* ptr = pZ;

	length_type n = size;

	vsum0 = S::zero();
	vsum1 = S::zero();
	vsum2 = S::zero();
	vsum3 = S::zero();

	while (n >= 4*S::vec_size)
	{
	  reg0 = S::load(ptr + 0*S::vec_size);
	  reg1 = S::load(ptr + 1*S::vec_size);
	  reg2 = S::load(ptr + 2*S::vec_size);
	  reg3 = S::load(ptr + 3*S::vec_size);
	  vsum0 = S::add(reg0, vsum0);
	  vsum1 = S::add(reg1, vsum1);
	  vsum2 = S::add(reg2, vsum2);
	  vsum3 = S::add(reg3, vsum3);
	  ptr += 4*S::vec_size;
	  n   -= 4*S::vec_size;
	}

	while (n >= S::vec_size)
	{
	  reg0 = S::load(ptr + 0*S::vec_size);
	  vsum0 = S::add(reg0, vsum0);
	  ptr += 1*S::vec_size;
	  n   -= 1*S::vec_size;
	}
      }
      t1.stop();

      vsum0 = S::add(vsum0, vsum1);
      vsum2 = S::add(vsum2, vsum3);
      vsum0 = S::add(vsum0, vsum2);

      S::store(psum, vsum0);
      res = psum[0];
      for (index_type i=1; i<S::vec_size; i+=1)
	res += psum[i];
    }
    
    test_assert(res == T(size*2));
    
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
  case   1: loop(t_sumval_simd<float>()); break;
  case   2: loop(t_sumval_simd_r4<float>()); break;
  case 101: loop(t_sumval_simd_no<float>()); break;
  case 102: loop(t_sumval_simd_r4_no<float>()); break;
  case   0:
    std::cout
      << "sumval_simd -- SIMD sumval\n"
      << "    -1 -- SIMD sumval (overhead included)\n"
      << "    -2 -- SIMD sumval (overhead not included)\n"
      << "  -101 -- SIMD sumval (overhead included, loop unrolled 4 times)\n"
      << "  -102 -- SIMD sumval (overhead not included, loop unrolled 4 times)\n"
      ;
  default: return 0;
  }
  return 1;
}
