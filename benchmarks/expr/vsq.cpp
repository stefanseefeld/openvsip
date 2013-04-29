//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector square

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/diagnostics.hpp>

#include "../benchmarks.hpp"

using namespace vsip;


/***********************************************************************
  Definitions - vector element-wise square
***********************************************************************/

template <typename T>
struct t_vsquare1 : Benchmark_base
{
  char const* what() { return "t_vsquare1"; }
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

    A.put(0, T(4));

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = sq(A);
    t1.stop();
    
    if (!equal(C.get(0), T(16)))
    {
      std::cout << "Oops!  C.get(0) is " << C.get(0) << "\n";
      std::cout << "t_vsquare1: ERROR" << std::endl;
      abort();
    }

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i)*A.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    vsip_csl::assign_diagnostics(C, sq(A));
  }
};


#ifdef VSIP_IMPL_SOURCERY_VPP
template <typename T, storage_format_type F>
struct t_vsquare2 : Benchmark_base
{
  // compile-time typedefs
  typedef Layout<1, row1_type, dense, F> LP;
  typedef impl::Strided<1, T, LP> block_type;

  // benchmark attributes
  char const* what() { return "t_vsquare2"; }
  int ops_per_point(length_type)  { return 1; }
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

    A.put(0, T(4));
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = sq(A);
    t1.stop();
    
    test_assert(equal(C.get(0), T(16)));

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i)*A.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> C(size);

    vsip_csl::assign_diagnostics(C, sq(A));
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
  case  1: loop(t_vsquare1<float>()); break;
  case  2: loop(t_vsquare1<complex<float> >()); break;
#ifdef VSIP_IMPL_SOURCERY_VPP
  case  3: loop(t_vsquare2<complex<float>, interleaved_complex>()); break;
  case  4: loop(t_vsquare2<complex<float>, split_complex>()); break;
#endif

  case  0:
    std::cout
      << "vsquare -- vector square\n"
      << "                F  - float\n"
      << "                CF - complex<float>\n"
      << "   -1 -- vector element-wise square -- F/F \n"
      << "   -2 -- vector element-wise square -- CF/CF\n"
      << "   -3 -- Vector<complex<float>> (INTER)\n"
      << "   -4 -- Vector<complex<float>> (SPLIT)\n"
      ;
  default:
    return 0;
  }
  return 1;
}
