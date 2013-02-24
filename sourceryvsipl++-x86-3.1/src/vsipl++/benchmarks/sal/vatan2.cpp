/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for SAL vector ATAN2.

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
struct t_vatan2_sal;

template <storage_format_type C>
struct t_vatan2_sal<float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vatan2_sal"; }
  int ops_per_point(size_t)  { return -1/*vsip::impl::Ops_info<float>::mul*/; }
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

    A = T(3);
    B = T(4);

    dda::Data<block_type, dda::in> ext_a(A.block());
    dda::Data<block_type, dda::in> ext_b(B.block());
    dda::Data<block_type, dda::out> ext_c(CC.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();
    T *pC = ext_c.ptr();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      // According to the Mercury SAL manual, this computes
      //   C[i] = atan( B[i] / A[i] )
      // The man page says atan2(b,a) == atan(b/a)
      vatan2x(const_cast<T*>(pA), 1, const_cast<T*>(pB), 1, pC, 1, size, 0 );
      // The dispatch converts
      //   C = atan2(A,B)
      // to the vatan2x call above.  But if our function is to
      // mimic the scalar, the call should be
      //   C = atan2(B,A)
      // and the dispatch should swap the parameters.
    t1.stop();
    
    if (!equal(pC[0], 0.927295f))
    {
      std::cout << "t_vatan2_sal: ERROR" << std::endl;
      std::cout << "t_vatan2_sal: ANS = " << pC[0] << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};

template <storage_format_type C>
struct t_vatan2_sal<double, C> : Benchmark_base
{
  typedef double T;

  char const *what() { return "t_vatan2_sal"; }
  int ops_per_point(size_t)  { return -1/*vsip::impl::Ops_info<double>::mul*/; }
  int riob_per_point(size_t) { return 2*sizeof(double); }
  int wiob_per_point(size_t) { return 1*sizeof(double); }
  int mem_per_point(size_t)  { return 3*sizeof(double); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   CC(size);

    A = T(3);
    B = T(4);

    dda::Data<block_type, dda::in> ext_a(A.block());
    dda::Data<block_type, dda::in> ext_b(B.block());
    dda::Data<block_type, dda::out> ext_c(CC.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();
    T *pC = ext_c.ptr();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      vatan2dx(const_cast<T*>(pA), 1, const_cast<T*>(pB), 1, pC, 1, size, 0 );
    t1.stop();
    
    if (!equal(pC[0], 0.927295))
    {
      std::cout << "t_vatan2_sal: ERROR" << std::endl;
      std::cout << "t_vatan2_sal: ANS = " << pC[0] << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};

template <typename T, storage_format_type C = interleaved_complex>
struct t_vatan2_sal_2;

template <storage_format_type C>
struct t_vatan2_sal_2<float, C> : Benchmark_base
{
  typedef float T;

  char const *what() { return "t_vatan2_sal_2"; }
  int ops_per_point(size_t)  { return -1/*vsip::impl::Ops_info<float>::mul*/; }
  int riob_per_point(size_t) { return 2*sizeof(float); }
  int wiob_per_point(size_t) { return 1*sizeof(float); }
  int mem_per_point(size_t)  { return 3*sizeof(float); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(2*size, T());
    Vector<T, block_type>   B(2*size, T());
    Vector<T, block_type>   CC(size);

    A = T(3);
    B = T(4);

    dda::Data<block_type, dda::in> ext_a(A.block());
    dda::Data<block_type, dda::in> ext_b(B.block());
    dda::Data<block_type, dda::out> ext_c(CC.block());
    
    T const *pA = ext_a.ptr();
    T const *pB = ext_b.ptr();
    T *pC = ext_c.ptr();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      // According to the Mercury SAL manual, this computes
      //   C[i] = atan( B[i] / A[i] )
      // The man page says atan2(b,a) == atan(b/a)
      vatan2x(const_cast<T*>(pA), 2, const_cast<T*>(pB), 2, pC, 1, size, 0 );
      // The dispatch converts
      //   C = atan2(A,B)
      // to the vatan2x call above.  But if our function is to
      // mimic the scalar, the call should be
      //   C = atan2(B,A)
      // and the dispatch should swap the parameters.
    t1.stop();
    
    if (!equal(pC[0], 0.927295f))
    {
      std::cout << "t_vatan2_sal_2: ERROR" << std::endl;
      std::cout << "t_vatan2_sal_2: ANS = " << pC[0] << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};

/* ----------------- */

template <typename T, storage_format_type C = interleaved_complex>
struct t_vatan2_sal_vpp : Benchmark_base
{
  char const *what() { return "t_vatan2_sal_vpp"; }
  int ops_per_point(size_t)  { return -1/*vsip::impl::Ops_info<T>::mul*/; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   CC(size);

    A = T(3);
    B = T(4);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      CC = atan2(B, A);
    t1.stop();
    
    if (!equal(CC(0), T(0.927295)))
    {
      std::cout << "t_vatan2_sal_vpp: ERROR" << std::endl;
      std::cout << "t_vatan2_sal_vpp: ANS = " << CC(0) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};

template <typename T, storage_format_type C = interleaved_complex>
struct t_vatan2_sal_vpp_2 : Benchmark_base
{
  char const *what() { return "t_vatan2_sal_vpp_2"; }
  int ops_per_point(size_t)  { return -1/*vsip::impl::Ops_info<T>::mul*/; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(2*size, T());
    Vector<T, block_type>   B(2*size, T());
    Vector<T, block_type>   CC(size);

    vsip::Domain<1> qDom(1, 2, size); 
    vsip::Domain<1> iDom(0, 2, size);

    A = T(3);
    B = T(4);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      CC = atan2(B(qDom), A(iDom));
    t1.stop();
    
    if (!equal(CC(0), T(0.927295)))
    {
      std::cout << "t_vatan2_sal_vpp_2: ERROR" << std::endl;
      std::cout << "t_vatan2_sal_vpp_2: ANS = " << CC(0) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }

  t_vatan2_sal_vpp_2(length_type thing=0) : thing_(thing) {}

  length_type thing_;
};

template <typename T, storage_format_type C = interleaved_complex>
struct t_vatan2_sal_scalar : Benchmark_base
{
  char const *what() { return "t_vatan2_sal_scalar"; }
  int ops_per_point(size_t)  { return -1/*vsip::impl::Ops_info<T>::mul*/; }
  int riob_per_point(size_t) { return 2*sizeof(T); }
  int wiob_per_point(size_t) { return 1*sizeof(T); }
  int mem_per_point(size_t)  { return 3*sizeof(T); }

  void operator()(size_t size, size_t loop, float& time)
  {
    typedef Layout<1, row1_type, dense, C> LP;
    typedef impl::Strided<1, T, LP, Local_map> block_type;

    Vector<T, block_type>   A(size, T());
    Vector<T, block_type>   B(size, T());
    Vector<T, block_type>   CC(size);

    A = T(3);
    B = T(4);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (size_t l=0; l<loop; ++l)
      for (index_type i=0; i<size; ++i)
	CC.put(i, atan2(B.get(i), A.get(i)));
    t1.stop();
    
    if (!equal(CC(0), T(0.927295)))
    {
      std::cout << "t_vatan2_sal_scalar: ERROR" << std::endl;
      std::cout << "t_vatan2_sal_scalar: ANS = " << CC(0) << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};





void
defaults(Loop1P& loop)
{
  loop.user_param_ = 0;
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vatan2_sal<float>()); break;
  case  2: loop(t_vatan2_sal<double>()); break;
  case  3: loop(t_vatan2_sal_2<float>()); break;

  case 11: loop(t_vatan2_sal_vpp<float>()); break;
  case 12: loop(t_vatan2_sal_vpp<double>()); break;
  case 13: loop(t_vatan2_sal_vpp_2<float>()); break;

  case 21: loop(t_vatan2_sal_scalar<float>()); break;

  default:
    std::cout << "   -1 : SAL vatan2x\n";
    std::cout << "   -2 : SAL vatan2dx\n";
    std::cout << "   -3 : SAL vatan2x, input strides 2\n";
    std::cout << "  -11 : C = atan2(A,B) {float}\n";
    std::cout << "  -12 : C = atan2(A,B) {double}\n";
    std::cout << "  -13 : C = atan2(A,B) {float}, input strides 2\n";
    std::cout << "  -21 : C = atan2(A,B) {float}, scalar\n";
    return 0;
  }
  return 1;
}
