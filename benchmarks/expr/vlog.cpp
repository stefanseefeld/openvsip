//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector log

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "../benchmark.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise log
***********************************************************************/

template <typename T>
struct t_vlog1 : Benchmark_base
{
  char const* what() { return "t_vlog1"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = log(A);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      if (!equal(C.get(i), log(A.get(i))))
      {
        std::cout << "Not equal at index " << i
	          << ", have " << C.get(i)
		  << ", should be " << log(A.get(i))
		  << ", input is " << A.get(i)
		  << "\n";
	test_assert(equal(C.get(i), log(A.get(i))));
      }
//    test_assert(equal(C.get(i), log(A.get(i))));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    std::cout << ovxx::assignment::diagnostics(C, log(A)) << std::endl;
  }
};

/***********************************************************************
  Definitions - vector element-wise log10
***********************************************************************/

template <typename T>
struct t_vlog101 : Benchmark_base
{
  char const* what() { return "t_vlog101"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      C = log10(A);
    time = t1.elapsed();
    
    for (index_type i=0; i<size; ++i)
      if (!equal(C.get(i), log10(A.get(i))))
      {
        std::cout << "Not equal at index " << i
	          << ", have " << C.get(i)
		  << ", should be " << log10(A.get(i))
		  << ", input is " << A.get(i)
		  << "\n";
	test_assert(equal(C.get(i), log10(A.get(i))));
      }
//    test_assert(equal(C.get(i), log(A.get(i))));
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);
    
    std::cout << ovxx::assignment::diagnostics(C, log10(A)) << std::endl;
  }
};

void defaults(Loop1P&) {}

int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_vlog1<float>()); break;
  case  2: loop(t_vlog1<complex<float> >()); break;
  case 11: loop(t_vlog101<float>()); break;
  case 12: loop(t_vlog101<complex<float> >()); break;

  case  0:
    std::cout
      << "vlog -- vector log\n"
      << "                F  - float\n"
      << "               CF  - complex\n"
      << "   -1 -- vector element-wise log    -- F/F \n"
      << "   -2 -- vector element-wise log    -- CF/CF \n"
      << "  -11 -- vector element-wise log10  -- F/F \n"
      << "  -12 -- vector element-wise log10  -- CF/CF \n"
      ;
  default:
    return 0;
  }
  return 1;
}
