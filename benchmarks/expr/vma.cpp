//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for fused multiply-add and variants such as axpy.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "../benchmark.hpp"


using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise fused multiply-add
***********************************************************************/

template <typename T1,
	  typename T2>
struct Ops2_info
{
  static unsigned int const div = 1;
  static unsigned int const sqr = 1;
  static unsigned int const mul = 1;
  static unsigned int const add = 1;
};



template <typename T>
struct Ops2_info<T, complex<T> >
{
  static unsigned int const div = 6 + 3 + 2;
  static unsigned int const mul = 2;
  static unsigned int const add = 1;
};



template <typename T>
struct Ops2_info<complex<T>, T >
{
  static unsigned int const div = 2;
  static unsigned int const mul = 2;
  static unsigned int const add = 1;
};

template <typename T>
struct Ops2_info<complex<T>, complex<T> >
{
  static unsigned int const div = 6 + 3 + 2;
  static unsigned int const mul = 4 + 2;
  static unsigned int const add = 2;
};



template <typename T,
	  dimension_type DimA,
	  dimension_type DimB,
	  dimension_type DimC>
struct t_vma : Benchmark_base
{
  char const* what() { return "t_vma"; }
  int ops_per_point(length_type)
    { return ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T> A(size, T(3));
    Vector<T> B(size, T(4));
    Vector<T> C(size, T(5));
    Vector<T>        X(size, T(0));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      X = A * B + C;
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(X.get(i), T(3*4+5)));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T> A(size, T(3));
    Vector<T> B(size, T(4));
    Vector<T> C(size, T(5));
    Vector<T>        X(size, T(0));

    std::cout << ovxx::assignment::diagnostics(X, A * B + C) << std::endl;
  }
};

template <typename T,
	  dimension_type DimA,
	  dimension_type DimB,
	  dimension_type DimC>
struct t_vma_fcn : Benchmark_base
{
  char const* what() { return "t_vma_fcn"; }
  int ops_per_point(length_type)
    { return ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T> A(size, T(3));
    Vector<T> B(size, T(4));
    Vector<T> C(size, T(5));
    Vector<T>        X(size, T(0));

    timer t1;
    for (index_type l=0; l<loop; ++l)
//    X = A * B + C;
      X = ma(A, B, C);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(X.get(i), T(3*4+5)));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T> A(size, T(3));
    Vector<T> B(size, T(4));
    Vector<T> C(size, T(5));
    Vector<T>        X(size, T(0));

    std::cout << ovxx::assignment::diagnostics(X, ma(A, B, C)) << std::endl;
  }
};



template <typename T,
	  dimension_type DimA,
	  dimension_type DimB,
	  dimension_type DimC>
struct t_vma_nonfused : Benchmark_base
{
  char const* what() { return "t_vma_nonfused"; }
  int ops_per_point(length_type)
    { return ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T> A(size, T(3));
    Vector<T> B(size, T(4));
    Vector<T> C(size, T(5));
    Vector<T>        tmp(size, T(0));
    Vector<T>        X(size, T(0));

    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      tmp = A * B;
      X   = tmp + C;
    }
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(X.get(i), T(3*4+5)));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T> A(size, T(3));
    Vector<T> B(size, T(4));
    Vector<T> C(size, T(5));
    Vector<T>        tmp(size, T(0));
    Vector<T>        X(size, T(0));

    std::cout << ovxx::assignment::diagnostics(tmp, A * B) << std::endl;
    std::cout << ovxx::assignment::diagnostics(X,  tmp + C) << std::endl;
  }
};



// In-place multiply-add, aka AXPY (Y = A*X + Y)

template <typename TA,
	  typename TB,
	  dimension_type DimA,
	  dimension_type DimB>
struct t_vma_ip : Benchmark_base
{
  typedef typename Promotion<TA, TB>::type T;

  char const* what() { return "t_vma_ip"; }
  int ops_per_point(length_type)
  { return Ops2_info<TA, TB>::mul + Ops2_info<T, T>::add; }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<TA> A(size, TA(0));
    Vector<TB> B(size, TB(0));
    Vector<T>         X(size, T(5));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      X += A * B;
    time = t1.elapsed();

    A = TA(3);
    B = TB(4);

    X += A * B;
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(X.get(i), T(3*4+5)));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<TA> A(size, TA(0));
    Vector<TB> B(size, TB(0));
    Vector<T>         X(size, T(5));

    std::cout << ovxx::assignment::diagnostics(X, X + A * B) << std::endl;
  }
};



// In-place multiply-add, aka AXPY (Y = A*X + Y)

template <typename TA,
	  typename TB,
	  dimension_type DimA,
	  dimension_type DimB>
struct t_vma_ip_nonfused : Benchmark_base
{
  typedef typename Promotion<TA, TB>::type T;

  char const* what() { return "t_vma_ip_nonfused"; }
  int ops_per_point(length_type)
  { return Ops2_info<TA, TB>::mul + Ops2_info<T, T>::add; }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<TA> A(size, TA(0));
    Vector<TB> B(size, TB(0));
    Vector<T>         tmp(size, T(0));
    Vector<T>         X(size, T(5));

    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      tmp = A * B;
      X   = X + tmp;
    }
    time = t1.elapsed();

    A = TA(3);
    B = TB(4);

    X += A * B;
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(X.get(i), T(3*4+5)));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<TA> A(size, TA(0));
    Vector<TB> B(size, TB(0));
    Vector<T>         X(size, T(5));

    std::cout << ovxx::assignment::diagnostics(X, X + A * B) << std::endl;
  }
};




template <typename T>
struct t_vma_cSC : Benchmark_base
{
  char const* what() { return "t_vma_cSC"; }
  int ops_per_point(length_type)
  {
    return Ops2_info<T, complex<T> >::mul +
	   Ops2_info<complex<T>, complex<T> >::add;
  }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    complex<T>          a = complex<T>(3, 0);
    Vector<T>           B(size, T(4));
    Vector<complex<T> > C(size, complex<T>(5, 0));
    Vector<complex<T> > X(size, complex<T>(0, 0));

    timer t1;
    for (index_type l=0; l<loop; ++l)
      X = a * B + C;
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(X.get(i), complex<T>(3*4+5, 0)));
  }

  void diag()
  {
    length_type const size = 256;

    complex<T>          a = complex<T>(3, 0);
    Vector<T>           B(size, T(4));
    Vector<complex<T> > C(size, complex<T>(5, 0));
    Vector<complex<T> > X(size, complex<T>(0, 0));

    std::cout << ovxx::assignment::diagnostics(X, a * B + C) << std::endl;
  }
};



#if DO_SIMD
template <typename T>
struct t_vma_cSC_simd : Benchmark_base
{
  char const* what() { return "t_vma_cSC_simd"; }
  int ops_per_point(length_type)
  {
    return Ops2_info<T, complex<T> >::mul +
	   Ops2_info<complex<T>, complex<T> >::add;
  }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    using vsip::dda::Data;

    typedef Dense<1, T>           sblock_type;
    typedef Dense<1, complex<T> > cblock_type;

    complex<T>          a = complex<T>(3, 0);
    Vector<T, sblock_type>          B(size, T(4));
    Vector<complex<T>, cblock_type> C(size, complex<T>(5, 0));
    Vector<complex<T>, cblock_type> X(size, complex<T>(0, 0));

    {
      dda::Data<sblock_type> B_ext(B.block());
      dda::Data<cblock_type> C_ext(C.block());
      dda::Data<cblock_type> X_ext(X.block());
    
      timer t1;
      for (index_type l=0; l<loop; ++l)
	vsip::impl::simd::vma_cSC(a, B_ext.ptr(), C_ext.ptr(), X_ext.ptr(),
				  size);
      time = t1.elapsed();
    }
    
    for (index_type i=0; i<size; ++i)
    {
      if (!equal(X.get(i), complex<T>(3*4+5, 0)))
	std::cout << i << " = " << X.get(i) << std::endl;
      test_assert(equal(X.get(i), complex<T>(3*4+5, 0)));
    }
  }

  void diag()
  {
    // using vsip::impl::simd::is_algorithm_supported;
    // using vsip::impl::simd::Alg_vma_cSC;

    // static bool const Is_vectorized =
    //   is_algorithm_supported<std::complex<T>, false, Alg_vma_cSC>::value;

    // std::cout << "is_vectorized: "
    // 	      << (Is_vectorized ? "yes" : "no")
    // 	      << std::endl;
  }
};
#endif

void defaults(Loop1P&) {}

int
benchmark(Loop1P& loop, int what)
{
  typedef float           SF;
  typedef complex<float>  CF;
  typedef double          SD;
  typedef complex<double> CD;

  switch (what)
  {
  case   1: loop(t_vma<SF, 1, 1, 1>()); break;
  case   2: loop(t_vma<SF, 0, 1, 1>()); break;
  case   3: loop(t_vma<SF, 0, 1, 0>()); break;

  case  11: loop(t_vma<CF, 1, 1, 1>()); break;
  case  12: loop(t_vma<CF, 0, 1, 1>()); break;
  case  13: loop(t_vma<CF, 0, 1, 0>()); break;

  case  21: loop(t_vma_ip<SF, SF, 1, 1>()); break;
  case  22: loop(t_vma_ip<SF, SF, 0, 1>()); break;

  case  31: loop(t_vma_ip<CF, CF, 1, 1>()); break;
  case  32: loop(t_vma_ip<CF, CF, 0, 1>()); break;

  case  41: loop(t_vma_ip<CF, SF, 0, 1>()); break;

  case  51: loop(t_vma_nonfused<CF, 1, 1, 1>()); break;
  case  52: loop(t_vma_ip_nonfused<CF, CF, 1, 1>()); break;

  case  61: loop(t_vma_fcn<SF, 1, 1, 1>()); break;
  case  62: loop(t_vma_fcn<SF, 0, 1, 1>()); break;
  case  63: loop(t_vma_fcn<SF, 0, 1, 0>()); break;

  case  71: loop(t_vma_fcn<CF, 1, 1, 1>()); break;
  case  72: loop(t_vma_fcn<CF, 0, 1, 1>()); break;
  case  73: loop(t_vma_fcn<CF, 0, 1, 0>()); break;

  case 141: loop(t_vma_ip<CD, SD, 0, 1>()); break;

  case 201: loop(t_vma_cSC<SF>()); break;
#if DO_SIMD
  case 202: loop(t_vma_cSC_simd<SF>()); break;
#endif
  case 203: loop(t_vma_cSC<SD>()); break;
#if DO_SIMD
  case 204: loop(t_vma_cSC_simd<SD>()); break;
#endif

  case 0:
    std::cout
      << "vma -- vector multiply-add\n"
      << "  -11 -- V = A * B + C [complex]\n"
      << "  -21 -- V += A * B [float]\n"
      << "  -22 -- V += a * B [float]\n"
      << "  -31 -- V += A * B [complex]\n"
      << "  -32 -- V += a * B [complex]\n"
      << "  -51 -- V = A * B + C [complex], nonfused\n"
      << "  -61 -- V = ma(A,B,C) [float]\n"
      << "  -71 -- V = ma(A,B,C) [complex]\n"
      << " -201 -- V = a * B + C [complex*float + complex]\n"
      << std::endl;
  default:
    return 0;
  }
  return 1;
}
