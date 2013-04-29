//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector cosine & sine

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/diagnostics.hpp>

#include "../benchmarks.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise cos
***********************************************************************/

template <typename T>
struct t_vcos1 : Benchmark_base
{
  char const* what() { return "t_vcos1"; }
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

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = cos(A);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), cos(A.get(i))));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    vsip_csl::assign_diagnostics(C, cos(A));
  }
};

/***********************************************************************
  Definitions - vector element-wise sin
***********************************************************************/

template <typename T>
struct t_vsin1 : Benchmark_base
{
  char const* what() { return "t_vsin1"; }
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

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = sin(A);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), sin(A.get(i))));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    vsip_csl::assign_diagnostics(C, sin(A));
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
  case  1: loop(t_vcos1<float>()); break;
  case  2: loop(t_vcos1<complex<float> >()); break;
  case 11: loop(t_vsin1<float>()); break;
  case 12: loop(t_vsin1<complex<float> >()); break;

  case  0:
    std::cout
      << "vcos -- vector cos & sin\n"
      << "                F  - float\n"
      << "               CF  - complex\n"
      << "   -1 -- vector element-wise cos  -- F/F \n"
      << "   -2 -- vector element-wise cos  -- CF/CF \n"
      << "  -11 -- vector element-wise sin  -- F/F \n"
      << "  -12 -- vector element-wise sin  -- CF/CF \n"
      ;
  default:
    return 0;
  }
  return 1;
}
