/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/vdiv.cpp
    @author  Don McCoy
    @date    2006-04-30
    @brief   VSIPL++ Library: Benchmark for vector divide.

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
  Definitions - vector element-wise divide
***********************************************************************/

template <typename T>
struct t_vdiv1 : Benchmark_base
{
  char const* what() { return "t_vdiv1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(6));
    B.put(0, T(3));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = A / B;
    t1.stop();
    
    if (!equal(C.get(0), T(2)))
    {
      std::cout << "t_vdiv1: ERROR" << std::endl;
      abort();
    }

    for (index_type i=0; i<size; ++i)
    {
      T want = A.get(i) / B.get(i);
      if (!equal(C.get(i), want))
      {
	std::cout << "t_vdiv1: ERROR at " << i << std::endl;
	std::cout << " Is " << C.get(i) << ", should be " << want << std::endl;
	static int oops = 0;
	++oops;
	if (oops > 10)
	  abort();
      }
#if 0
      test_assert(equal(C.get(i), A.get(i) / B.get(i)));
#else
    }
#endif
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    impl::assign_diagnostics(C, A / B);
  }
};



template <typename T>
struct t_vdiv_ip1 : Benchmark_base
{
  char const* what() { return "t_vdiv_ip1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T(1));
    Vector<T>   C(size);
    Vector<T>   chk(size);

    Rand<T> gen(0, 0);
    chk = gen.randu(size);
    C = chk;

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C /= A;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(chk.get(i), C.get(i)));
    
    time = t1.delta();
  }
};



template <typename T>
struct t_vdiv_dom1 : Benchmark_base
{
  char const* what() { return "t_vdiv_dom1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(6));
    B.put(0, T(3));

    Domain<1> dom(size);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C(dom) = A(dom) / B(dom);
    t1.stop();
    
    if (!equal(C.get(0), T(2)))
    {
      std::cout << "t_vdiv_dom1: ERROR" << std::endl;
      abort();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) / B.get(i)));
    
    time = t1.delta();
  }
};


#ifdef VSIP_IMPL_SOURCERY_VPP
template <typename T, typename ComplexFmt>
struct t_vdiv2 : Benchmark_base
{
  char const* what() { return "t_vdiv2"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef impl::Layout<1, row1_type, impl::Stride_unit_dense, ComplexFmt>
		LP;
    typedef impl::Strided<1, T, LP> block_type;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    A.put(0, T(6));
    B.put(0, T(3));
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = A / B;
    t1.stop();
    
    if (!equal(C.get(0), T(2)))
    {
      std::cout << "t_vdiv2: ERROR" << std::endl;
      abort();
    }
    
    time = t1.delta();
  }
};
#endif // VSIP_IMPL_SOURCERY_VPP


/***********************************************************************
  Definitions - real / complex vector element-wise divide
***********************************************************************/

template <typename T>
struct t_rcvdiv1 : Benchmark_base
{
  char const* what() { return "t_rcvdiv1"; }
  int ops_per_point(length_type)  { return 2*vsip::impl::Ops_info<T>::div; }
  int riob_per_point(length_type) { return sizeof(T) + sizeof(complex<T>); }
  int wiob_per_point(length_type) { return 1*sizeof(complex<T>); }
  int mem_per_point(length_type)  { return 1*sizeof(T)+2*sizeof(complex<T>); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<complex<T> > A(size);
    Vector<T>           B(size);
    Vector<complex<T> > C(size);

    Rand<complex<T> > cgen(0, 0);
    Rand<T>           sgen(0, 0);

    A = cgen.randu(size);
    B = sgen.randu(size);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = B / A;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), B.get(i) / A.get(i)));
    
    time = t1.delta();
  }
};



// Benchmark scalar-view vector divide (Scalar / View)

template <typename ScalarT,
	  typename T>
struct t_svdiv1 : Benchmark_base
{
  char const* what() { return "t_svdiv1"; }
  int ops_per_point(length_type)
  { if (sizeof(ScalarT) == sizeof(T))
      return vsip::impl::Ops_info<T>::div;
    else
      return 2*vsip::impl::Ops_info<ScalarT>::div;
  }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T(1));
    Vector<T>   C(size);

    ScalarT alpha = ScalarT(3);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = alpha / A;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), alpha / A.get(i)));
    
    time = t1.delta();
  }
};



// Benchmark scalar-view vector divide (Scalar / View)

template <typename T>
struct t_svdiv2 : Benchmark_base
{
  char const* what() { return "t_svdiv2"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::div; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    T alpha = T(3);

    A.put(0, T(6));
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = A / alpha;
    t1.stop();

    test_assert(equal(C.get(0), T(2)));
    
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
  case  1: loop(t_vdiv1<float>()); break;
  case  2: loop(t_vdiv1<complex<float> >()); break;
#ifdef VSIP_IMPL_SOURCERY_VPP
  case  3: loop(t_vdiv2<complex<float>, impl::Cmplx_inter_fmt>()); break;
  case  4: loop(t_vdiv2<complex<float>, impl::Cmplx_split_fmt>()); break;
#endif
  case  5: loop(t_rcvdiv1<float>()); break;

  case 11: loop(t_svdiv1<float,          float>()); break;
  case 12: loop(t_svdiv1<float,          complex<float> >()); break;
  case 13: loop(t_svdiv1<complex<float>, complex<float> >()); break;

  case 14: loop(t_svdiv2<float>()); break;
  case 15: loop(t_svdiv2<complex<float> >()); break;

  case 21: loop(t_vdiv_dom1<float>()); break;
  case 22: loop(t_vdiv_dom1<complex<float> >()); break;

  case 31: loop(t_vdiv_ip1<float>()); break;
  case 32: loop(t_vdiv_ip1<complex<float> >()); break;

  case  0:
    std::cout
      << "vdiv -- vector divide\n"
      << "                F  - float\n"
      << "                CF - complex<float>\n"
      << "   -1 -- vector element-wise divide -- F/F \n"
      << "   -2 -- vector element-wise divide -- CF/CF\n"
      << "   -5 -- real / complex vector element-wise divide\n"
      << "  -11 -- scalar-view vector divide (Scalar / View) -- F/F \n"
      << "  -12 -- scalar-view vector divide (Scalar / View) -- F/CF\n"
      << "  -13 -- scalar-view vector divide (Scalar / View) -- CF/CF\n"
      << "  -14 -- scalar-view vector divide (View / Scalar) -- F/F \n"
      << "  -15 -- scalar-view vector divide (View / Scalar) -- CF/CF\n"
      << "  -21 -- vector element-wise divide (using domains) -- F/F \n"
      << "  -22 -- vector element-wise divide (using domains) -- CF/CF\n"
      << "  -31 -- vector element-wise divide (in place) -- F/F \n"
      << "  -32 -- vector element-wise divide (in place) -- CF/CF\n"
      ;
  default:
    return 0;
  }
  return 1;
}
