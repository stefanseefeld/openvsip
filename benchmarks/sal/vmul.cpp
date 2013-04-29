//
// Copyright (c) 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for SAL vector multiply.

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
struct t_vmul_sal;

template <int X, typename T>
struct t_vmul_sal_ip;

template <storage_format_type C>
struct t_vmul_sal<float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vmul_sal"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<float>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   CC(size);

    A(0) = T(3);
    B(0) = T(4);

    dda::Data<block_type, dda::in> ext_a(A.block());
    dda::Data<block_type, dda::in> ext_b(B.block());
    dda::Data<block_type, dda::out> ext_c(CC.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();
    T *pC = ext_c.ptr();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      vmulx(const_cast<T*>(pA), 1, const_cast<T*>(pB), 1, pC, 1, size, 0);
    t1.stop();
    
    if (pC[0] != 12.f)
    {
      std::cout << "t_vmul_sal: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



template <>
struct t_vmul_sal<complex<float>, interleaved_complex> : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vmul_sal complex<float> inter"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B = gen.randu(size); B(0) = T(4);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
    
      T const *pA = ext_a.ptr();
      T const *pB = ext_b.ptr();
      T *pC = ext_c.ptr();
    
      int conj_flag = 1;  // don't conjugate
      t1.start();
      for (size_t l=0; l<loop; ++l)
	cvmulx((COMPLEX *)pA, 2, (COMPLEX *)pB, 2, 
	       (COMPLEX *)pC, 2, size, conj_flag, 0);
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



template <>
struct t_vmul_sal<complex<float>, split_complex> : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vmul_sal complex<float> split"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, split_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;
    typedef dda::Data<block_type, dda::out>::ptr_type ptr_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B = gen.randu(size); B(0) = T(4);

    vsip_csl::profile::Timer t1;
    
    {
      dda::Data<block_type, dda::inout> ext_a(A.block());
      dda::Data<block_type, dda::inout> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
    
      ptr_type pA = ext_a.ptr();
      ptr_type pB = ext_b.ptr();
      ptr_type pC = ext_c.ptr();

    int conj_flag = 1;  // don't conjugate
    t1.start();
    for (size_t l=0; l<loop; ++l)
      zvmulx((COMPLEX_SPLIT*)&pA, 1,
	     (COMPLEX_SPLIT*)&pB, 1, 
	     (COMPLEX_SPLIT*)&pC, 1, size, conj_flag, 0 );
    t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



template <>
struct t_vmul_sal_ip<1, float> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vmul_sal_ip float"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T(1));

    A(0) = T(3);

    dda::Data<block_type, dda::inout> ext_a(A.block());
    dda::Data<block_type, dda::inout> ext_b(B.block());
    
    T* pA = ext_a.ptr();
    T* pB = ext_b.ptr();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      vmulx( pA, 1, pB, 1, pA, 1, size, 0 );
    t1.stop();
    
    if (pA[0] != 3.f)
    {
      std::cout << "t_vmul_sal: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



template <>
struct t_vmul_sal_ip<1, complex<float> > : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vmul_sal complex<float>"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());

    A(0) = T(1, 0);
    B(0) = T(1, 0);

    dda::Data<block_type, dda::in> ext_a(A.block());
    dda::Data<block_type, dda::in> ext_b(B.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();

    vsip_csl::profile::Timer t1;
    
    int conj_flag = 1;  // don't conjugate
    t1.start();
    for (size_t l=0; l<loop; ++l)
      cvmulx( (COMPLEX *)pA, 2, (COMPLEX *)pB, 2, 
                                (COMPLEX *)pB, 2, size, conj_flag, 0 );
    t1.stop();
    
    if (pB[0].real() != 1.f)
    {
      std::cout << "t_vmul_sal: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



template <>
struct t_vmul_sal_ip<2, complex<float> > : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vmul_sal complex<float>"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());

    A(0) = T(1, 0);
    B(0) = T(1, 0);

    dda::Data<block_type, dda::inout> ext_a(A.block());
    dda::Data<block_type, dda::inout> ext_b(B.block());
    
    T* pA = ext_a.ptr();
    T* pB = ext_b.ptr();

    vsip_csl::profile::Timer t1;
    
    int conj_flag = 1;  // don't conjugate
    t1.start();
    for (size_t l=0; l<loop; ++l)
      cvmulx( (COMPLEX *)pA, 2, (COMPLEX *)pB, 2, 
                                (COMPLEX *)pA, 2, size, conj_flag, 0 );
    t1.stop();
    
    if (pB[0].real() != 1.f)
    {
      std::cout << "t_vmul_sal: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};



/***********************************************************************
  Definitions - scalar-vector element-wise multiply
***********************************************************************/

template <typename ScalarT, typename T, storage_format_type C = interleaved_complex>
struct t_svmul_sal;

template <storage_format_type C>
struct t_svmul_sal<float, float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_svmul_sal"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<float>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(1, T());
    Vector<T, block_type>   CC(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B(0) = T(4);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(CC.block());
    
      T const *pA = ext_a.ptr();
      T const *pB = ext_b.ptr();
      T* pC = ext_c.ptr();
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
	vsmulx(const_cast<T*>(pA), 1, const_cast<T*>(pB), pC, 1, size, 0);
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(CC.get(i), A.get(i) * B.get(0)));
    
    time = t1.delta();
  }
};



template <>
struct t_svmul_sal<complex<float>, complex<float>, interleaved_complex>
  : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_svmul_sal"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, interleaved_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(1, T());
    Vector<T, block_type>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B(0) = T(4);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::in> ext_a(A.block());
      dda::Data<block_type, dda::in> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();
    T *pC = ext_c.ptr();
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      cvcsmlx((COMPLEX*)pA, 2,
	      (COMPLEX*)pB,
	      (COMPLEX*)pC, 2,
	      size, 0 );
    t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(0)));
    
    time = t1.delta();
  }
};



template <>
struct t_svmul_sal<complex<float>, complex<float>, split_complex>
  : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_svmul_sal"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, split_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;
    typedef dda::Data<block_type, dda::inout>::ptr_type ptr_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(1, T());
    Vector<T, block_type>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size); A(0) = T(3);
    B(0) = T(4);

    vsip_csl::profile::Timer t1;

    {
      dda::Data<block_type, dda::inout> ext_a(A.block());
      dda::Data<block_type, dda::inout> ext_b(B.block());
      dda::Data<block_type, dda::out> ext_c(C.block());
    
      ptr_type pA = ext_a.ptr();
      ptr_type pB = ext_b.ptr();
      ptr_type pC = ext_c.ptr();
    
      t1.start();
      for (size_t l=0; l<loop; ++l)
	zvzsmlx((COMPLEX_SPLIT*)&pA, 1,
		(COMPLEX_SPLIT*)&pB,
		(COMPLEX_SPLIT*)&pC, 1,
		size, 0 );
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(0)));
    
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
  case  1: loop(t_vmul_sal<float>()); break;
  case  2: loop(t_vmul_sal<complex<float>, interleaved_complex>()); break;
  case  3: loop(t_vmul_sal<complex<float>, split_complex>()); break;

  case 11: loop(t_svmul_sal<float, float>()); break;
  case 13: loop(t_svmul_sal<cf_type, cf_type, interleaved_complex>()); break;
  case 14: loop(t_svmul_sal<cf_type, cf_type, split_complex>()); break;

  case 31: loop(t_vmul_sal_ip<1, float>()); break;
  case 32: loop(t_vmul_sal_ip<1, complex<float> >()); break;
  case 33: loop(t_vmul_sal_ip<2, complex<float> >()); break;
  case  0:
    std::cout
      << "vmul -- SAL vector multiply\n"
      << "   -1: vmulx\n"
      << "   -2: cvmulx\n"
      << "   -3: zvmulx\n"
      << "\n"
      << "  -11: vsmulx\n"
      << "  -13: cvsmulx\n"
      << "  -14: zvsmulx\n"
      << "\n"
      << "  -31: vmulx\n"
      << "  -32: cvmulx B = A * B \n"
      << "  -33: cvmulx A = A * B\n"
      ;
  default:
    return 0;
  }
  return 1;
}
