/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for IPP vector multiply.

#include <iostream>
#include <complex>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/profile.hpp>
#include <vsip/core/ops_info.hpp>

#include <ipps.h>

#include "loop.hpp"
#include "benchmarks.hpp"

using namespace vsip;

template <typename T, storage_format_type C = interleaved_complex>
struct t_vmul_ipp_vm;

template <storage_format_type CF>
struct t_vmul_ipp_vm<float, CF> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vmul_ipp_vm"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<float>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, CF> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   C(size);

    A(0) = T(3);
    B(0) = T(4);

    dda::Data<block_type, dda::in> ext_a(A.block());
    dda::Data<block_type, dda::in> ext_b(B.block());
    dda::Data<block_type, dda::out> ext_c(C.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();
    T* pC = ext_c.ptr();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      ippsMul_32f(pA, pB, pC, size);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



template <>
struct t_vmul_ipp_vm<complex<float>, interleaved_complex> : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vmul_ipp_vm complex<float> inter"; }
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
      T* pC = ext_c.ptr();
    
      int conj_flag = 1;  // don't conjugate
      t1.start();
      for (size_t l=0; l<loop; ++l)
	ippsMul_32fc((Ipp32fc*)pA, (Ipp32fc*)pB, (Ipp32fc*)pC, size);
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



#if 0
// IPP Does not provide split complex multiply.
template <>
struct t_vmul_ipp_vm<complex<float>, split_complex> : Benchmark_base
{
  typedef complex<float> T;

  char const *what() { return "t_vmul_ipp_vm complex<float> split"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, split_complex> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;
    typedef dda::Data<block_type, dda::out>::ptr_type ptr_type;
    typedef dda::Data<block_type, dda::in>::ptr_type const_ptr_type;

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
    
      const_ptr_type pA = ext_a.ptr();
      const_ptr_type pB = ext_b.ptr();
      ptr_type pC = ext_c.ptr();

      int conj_flag = 1;  // don't conjugate
      t1.start();
      for (size_t l=0; l<loop; ++l)
	; // NO IPP split complex multiply
      t1.stop();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};
#endif



/***********************************************************************
  IPP Vector Multiply (IPP memory allocation)
***********************************************************************/

template <typename T>
struct t_vmul_ipp;

template <int X, typename T>
struct t_vmul_ipp_ip;

template <>
struct t_vmul_ipp<float> : Benchmark_base
{
  typedef float T;
  char const *what() { return "t_vmul_ipp"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32f* A = ippsMalloc_32f(size);
    Ipp32f* B = ippsMalloc_32f(size);
    Ipp32f* C = ippsMalloc_32f(size);

    if (!A || !B || !C) throw(std::bad_alloc());

    for (size_t i=0; i<size; ++i)
    {
      A[i] = 0.f;
      B[i] = 0.f;
    }

    A[0] = 3.f;
    B[0] = 4.f;
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      ippsMul_32f(A, B, C, size);
    t1.stop();
    
    if (C[0] != 12.f)
    {
      std::cout << "t_vmul_ipp: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();

    ippsFree(C);
    ippsFree(B);
    ippsFree(A);
  }
};



template <>
struct t_vmul_ipp<std::complex<float> > : Benchmark_base
{
  typedef std::complex<float> T;

  char const *what() { return "t_vmul_ipp complex<float>"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32fc* A = ippsMalloc_32fc(size);
    Ipp32fc* B = ippsMalloc_32fc(size);
    Ipp32fc* C = ippsMalloc_32fc(size);

    if (!A || !B || !C) throw(std::bad_alloc());

    for (size_t i=0; i<size; ++i)
    {
      A[i].re = 0.f;
      A[i].im = 0.f;
      B[i].re = 0.f;
      B[i].im = 0.f;
    }

    A[0].re = 3.f;
    B[0].re = 4.f;
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      ippsMul_32fc(A, B, C, size);
    t1.stop();
    
    if (C[0].re != 12.f)
    {
      std::cout << "t_vmul_ipp: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();

    ippsFree(C);
    ippsFree(B);
    ippsFree(A);
  }
};



template <>
struct t_vmul_ipp_ip<1, std::complex<float> > : Benchmark_base
{
  typedef std::complex<float> T;

  char const *what() { return "t_vmul_ipp complex<float>"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32fc* A = ippsMalloc_32fc(size);
    Ipp32fc* B = ippsMalloc_32fc(size);

    if (!A || !B) throw(std::bad_alloc());

    for (size_t i=0; i<size; ++i)
    {
      A[i].re = 0.f;
      A[i].im = 0.f;
      B[i].re = 0.f;
      B[i].im = 0.f;
    }

    A[0].re = 1.f;
    B[0].re = 3.f;
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      ippsMul_32fc(A, B, B, size);
    t1.stop();
    
    if (B[0].re != 3.f)
    {
      std::cout << "t_vmul_ipp: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();

    ippsFree(B);
    ippsFree(A);
  }
};



template <>
struct t_vmul_ipp_ip<2, std::complex<float> > : Benchmark_base
{
  typedef std::complex<float> T;

  char const *what() { return "t_vmul_ipp complex<float>"; }
  int ops_per_point(size_t)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 2*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    Ipp32fc* A = ippsMalloc_32fc(size);
    Ipp32fc* B = ippsMalloc_32fc(size);

    if (!A || !B) throw(std::bad_alloc());

    for (size_t i=0; i<size; ++i)
    {
      A[i].re = 0.f;
      A[i].im = 0.f;
      B[i].re = 0.f;
      B[i].im = 0.f;
    }

    A[0].re = 1.f;
    B[0].re = 3.f;
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      ippsMul_32fc_I(A, B, size);
    t1.stop();
    
    if (B[0].re != 3.f)
    {
      std::cout << "t_vmul_ipp: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();

    ippsFree(B);
    ippsFree(A);
  }
};



/***********************************************************************
  Benchmark Main
***********************************************************************/

void
defaults(Loop1P&)
{
}



void
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vmul_ipp_vm<float>()); break;
  case  2: loop(t_vmul_ipp_vm<std::complex<float> >()); break;

  case  12: loop(t_vmul_ipp_ip<1, std::complex<float> >()); break;
  case  22: loop(t_vmul_ipp_ip<2, std::complex<float> >()); break;

  case 101: loop(t_vmul_ipp<float>()); break;
  case 102: loop(t_vmul_ipp<std::complex<float> >()); break;

  case   0:
    std::cout
      << "vmul -- IPP vector multiply\n"
      << "\n"
      << "    -1: float           VSIPL++ allocation\n"
      << "    -2: complex<float>  VSIPL++ allocation\n"
      << "\n"
      << "   -12: complex<float>  IPP allocation (ippsMul_32fc)   (in place)\n"
      << "   -22: complex<float>  IPP allocation (ippsMul_32fc_I) (in place)\n"
      << "\n"
      << "  -101: float           IPP allocation (ippsMul_32f)\n"
      << "  -102: complex<float>  IPP allocation (ippsMul_32fc)vmul -- IPP vector multiply\n"
      ;
  }
}
