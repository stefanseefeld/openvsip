//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for SAL vector multiply-add.

#include <iostream>
#include <complex>

#include <vsip/random.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/core/ops_info.hpp>
#include <sal.h>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace vsip;

template <typename T, storage_format_type C = interleaved_complex>
struct t_vma_sal;

template <storage_format_type C>
struct t_vma_sal<float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vma_sal"; }
  int ops_per_point(size_t)  
    { return vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add; }
  int riob_per_point(size_t) { return 3*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 4*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   CC(size, T());
    Vector<T, block_type>   Z(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B = gen.randu(size); B(0) = T(4);
    CC = gen.randu(size); CC(0) = T(5);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::in> ext_c(CC.block());
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T const *pA = ext_a.ptr();
      T const *pB = ext_b.ptr();
      T const *pC = ext_c.ptr();
      T* pZ = ext_z.ptr();

    
      t1.start();
      for (size_t l=0; l<loop; ++l)
	vma_x(const_cast<T*>(pA), 1, const_cast<T*>(pB), 1, const_cast<T*>(pC), 1, pZ, 1, size, 0 );
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(Z.get(i), A.get(i) * B.get(i) + CC.get(i)));
    
    time = t1.delta();
  }
};



// vsma: (scalar * vector) + vector
template <typename T, storage_format_type C = interleaved_complex>
struct t_vsma_sal;

template <storage_format_type C>
struct t_vsma_sal<float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vsma_sal"; }
  int ops_per_point(size_t)  
    { return vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    T                       b;
    Vector<T, block_type>   CC(size, T());
    Vector<T, block_type>   Z(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    b = T(4);
    CC = gen.randu(size); CC(0) = T(5);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_c(CC.block());
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T const *pA = ext_a.ptr();
      T const *pB = &b;
      T const *pC = ext_c.ptr();
      T *pZ = ext_z.ptr();

      
      t1.start();
      for (size_t l=0; l<loop; ++l)
	vsmax(const_cast<T*>(pA), 1, const_cast<T*>(pB), const_cast<T*>(pC), 1, pZ, 1, size, 0);
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(Z.get(i), A.get(i) * b + CC.get(i)));
    
    time = t1.delta();
  }
};



template <>
struct t_vsma_sal<complex<float>, interleaved_complex> : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vsma_sal complex<float> inter"; }
  int ops_per_point(size_t)  
    { return vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(1, T());
    Vector<T, block_type>   C(size, T());
    Vector<T, block_type>   Z(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B(0) = T(4);
    C = gen.randu(size); C(0) = T(5);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::in> ext_c(C.block());
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      T const *pA = ext_a.ptr();
      T const *pB = ext_b.ptr();
      T const *pC = ext_c.ptr();
      T* pZ = ext_z.ptr();
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
	cvsmax((COMPLEX*)pA, 2,
	       (COMPLEX*)pB, 
	       (COMPLEX*)pC, 2,
	       (COMPLEX*)pZ, 2,
	       size, 0 );
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(Z.get(i), A.get(i) * B.get(0) + C.get(i)));
    
    time = t1.delta();
  }
};



template <>
struct t_vsma_sal<complex<float>, split_complex> : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vsma_sal complex<float> split"; }
  int ops_per_point(size_t)  
    { return vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, split_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;
    typedef dda::Data<block_type, dda::in>::ptr_type ptr_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size, T());
    Vector<T, block_type>   Z(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
                         B(0) = T(4);
    C = gen.randu(size); C(0) = T(5);

    vsip_csl::profile::Timer t1;
    
    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::in> ext_c(C.block());
      dda::Data<block_type, dda::out> ext_z(Z.block());
    
      ptr_type pA = ext_a.ptr();
      ptr_type pB = ext_b.ptr();
      ptr_type pC = ext_c.ptr();
      ptr_type pZ = ext_z.ptr();
      
      t1.start();
      for (size_t l=0; l<loop; ++l)
	zvsmax((COMPLEX_SPLIT*)&pA, 1,
	       (COMPLEX_SPLIT*)&pB,
	       (COMPLEX_SPLIT*)&pC, 1, 
	       (COMPLEX_SPLIT*)&pZ, 1, size, 0 );
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(Z.get(i), A.get(i) * B.get(0) + C.get(i)));
    
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
  case  1: loop(t_vma_sal<float>()); break;

  case  11: loop(t_vsma_sal<float>()); break;
  case  12: loop(t_vsma_sal<complex<float>, interleaved_complex>()); break;
  case  13: loop(t_vsma_sal<complex<float>, split_complex>()); break;
#if 0
  case  2: loop(t_vmul_sal<complex<float> >()); break;

  case  12: loop(t_vmul_sal_ip<1, complex<float> >()); break;
  case  22: loop(t_vmul_sal_ip<2, complex<float> >()); break;
#endif
  case 0:
    std::cout
      << "vma -- SAL vector multiply-add\n"
      << "    -1: vma_x\n"
      << "   -11: vsmax\n"
      << "   -12: cvsmax\n"
      << "   -13: zvsmax\n"
      ;
  default:
    return 0;
  }
  return 1;
}
