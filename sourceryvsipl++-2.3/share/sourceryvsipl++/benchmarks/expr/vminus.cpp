/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/expr/vminus.cpp
    @author  Mike LeBlanc
    @date    2009-10-04
    @brief   VSIPL++ Library: Benchmark for vector minus

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/assign_diagnostics.hpp>

#include "../benchmarks.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise minus
***********************************************************************/

template <typename T>
struct t_vminus1 : Benchmark_base
{
  char const* what() { return "t_vminus1"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    A.put(0, T(16));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = -A;
    t1.stop();
    
    if (!equal(C.get(0), -T(16)))
    {
      std::cout << "Oops!  C.get(0) is " << C.get(0) << "\n";
      std::cout << "t_vminus1: ERROR" << std::endl;
      abort();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), -A.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    impl::assign_diagnostics(C, -A);
  }
};


#ifdef VSIP_IMPL_SOURCERY_VPP
template <typename T, typename ComplexFmt>
struct t_vminus2 : Benchmark_base
{
  // compile-time typedefs
  typedef impl::Layout<1, row1_type, impl::Stride_unit_dense, ComplexFmt>
		LP;
  typedef impl::Strided<1, T, LP> block_type;

  // benchmark attributes
  char const* what() { return "t_vminus2"; }
  int ops_per_point(length_type)  { return 2; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T, block_type> A(size, T());
    Vector<T, block_type> C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    A.put(0, T(3));
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = -A;
    t1.stop();
    
    for (index_type i=0; i!=size; ++i)
      if (!equal(C.get(i), -A.get(i)))
      {
	std::cout << "Check: C.get(" << i << ") is " << C.get(i) << "\n";
	break;
      }

    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> C(size);

    impl::assign_diagnostics(C, -A);
  }
};
#endif // VSIP_IMPL_SOURCERY_VPP






void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vminus1<float>()); break;
  case  2: loop(t_vminus1<complex<float> >()); break;
#ifdef VSIP_IMPL_SOURCERY_VPP
  case  3: loop(t_vminus2<complex<float>, impl::Cmplx_inter_fmt>()); break;
  case  4: loop(t_vminus2<complex<float>, impl::Cmplx_split_fmt>()); break;
#endif

  case  0:
    std::cout
      << "vminus -- vector minus\n"
      << "                F  - float\n"
      << "                CF - complex<float>\n"
      << "   -1 -- vector element-wise minus -- F/F \n"
      << "   -2 -- vector element-wise minus -- CF/CF\n"
      << "   -3 -- Vector<complex<float>> (INTER)\n"
      << "   -4 -- Vector<complex<float>> (SPLIT)\n"
      ;
  default:
    return 0;
  }
  return 1;
}
