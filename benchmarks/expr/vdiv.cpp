//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector divide.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "../benchmark.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise divide
***********************************************************************/

template <typename T>
struct t_vdiv1 : Benchmark_base
{
  char const* what() { return "t_vdiv1"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(6));
    B.put(0, T(3));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = A / B;
    time = t1.elapsed();
    
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
  }

  void diag()
  {
    using ovxx::assignment::diagnostics;

    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    std::cout << diagnostics(C, A / B) << std::endl;
  }
};



template <typename T>
struct t_vdiv_ip1 : Benchmark_base
{
  char const* what() { return "t_vdiv_ip1"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>   A(size, T(1));
    Vector<T>   C(size);
    Vector<T>   chk(size);

    Rand<T> gen(0, 0);
    chk = gen.randu(size);
    C = chk;

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C /= A;
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(chk.get(i), C.get(i)));
  }
};



template <typename T>
struct t_vdiv_dom1 : Benchmark_base
{
  char const* what() { return "t_vdiv_dom1"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
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
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      C(dom) = A(dom) / B(dom);
    time = t1.elapsed();
    
    if (!equal(C.get(0), T(2)))
    {
      std::cout << "t_vdiv_dom1: ERROR" << std::endl;
      abort();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) / B.get(i)));
  }
};


#ifdef VSIP_IMPL_SOURCERY_VPP
template <typename T, vsip::storage_format_type F>
struct t_vdiv2 : Benchmark_base
{
  char const* what() { return "t_vdiv2"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::div; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    typedef Layout<1, row1_type, dense, F> LP;
    typedef impl::Strided<1, T, LP> block_type;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    A.put(0, T(6));
    B.put(0, T(3));
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = A / B;
    time = t1.elapsed();
    
    if (!equal(C.get(0), T(2)))
    {
      std::cout << "t_vdiv2: ERROR" << std::endl;
      abort();
    }
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
  int ops_per_point(length_type)  { return 2*ovxx::ops_count::traits<T>::div; }
  int riob_per_point(length_type) { return sizeof(T) + sizeof(complex<T>); }
  int wiob_per_point(length_type) { return 1*sizeof(complex<T>); }
  int mem_per_point(length_type)  { return 1*sizeof(T)+2*sizeof(complex<T>); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<complex<T> > A(size);
    Vector<T>           B(size);
    Vector<complex<T> > C(size);

    Rand<complex<T> > cgen(0, 0);
    Rand<T>           sgen(0, 0);

    A = cgen.randu(size);
    B = sgen.randu(size);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = B / A;
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), B.get(i) / A.get(i)));
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
      return ovxx::ops_count::traits<T>::div;
    else
      return 2*ovxx::ops_count::traits<ScalarT>::div;
  }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T(1));
    Vector<T>   C(size);

    ScalarT alpha = ScalarT(3);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = alpha / A;
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), alpha / A.get(i)));
  }
};



// Benchmark scalar-view vector divide (Scalar / View)

template <typename T>
struct t_svdiv2 : Benchmark_base
{
  char const* what() { return "t_svdiv2"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::div; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    T alpha = T(3);

    A.put(0, T(6));
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = A / alpha;
    time = t1.elapsed();

    test_assert(equal(C.get(0), T(2)));
  }
};

void defaults(Loop1P&) {}

int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vdiv1<float>()); break;
  case  2: loop(t_vdiv1<complex<float> >()); break;
#ifdef VSIP_IMPL_SOURCERY_VPP
  case  3: loop(t_vdiv2<complex<float>, interleaved_complex>()); break;
  case  4: loop(t_vdiv2<complex<float>, split_complex>()); break;
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
