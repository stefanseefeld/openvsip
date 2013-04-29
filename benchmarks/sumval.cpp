//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for sumval reductions.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/math.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/profile.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

#include "benchmarks.hpp"

using namespace vsip;
using namespace vsip_csl;



/***********************************************************************
  VSIPL++ sumval
***********************************************************************/

template <typename T>
struct t_sumval1 : Benchmark_base
{
  char const* what() { return "t_sumval_vector"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   view(size, T());
    T           val = T();
    
    Rand<T>     gen(0, 0);

    if (init_ == 0)
      view = gen.randu(size);
    else if (init_ == 1)
      view(0) = T(2);
    else if (init_ == 2)
      view(size-1) = T(2);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = vsip::sumval(view);
    t1.stop();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, T(2)));
    
    time = t1.delta();
  }

  t_sumval1(int init) : init_(init) {}

  // member data.
  int init_;
};

template <typename T, typename R>
struct t_sumval1_ext_base : Benchmark_base
{
  char const* what() { return "t_sumval_ext_vector"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   view(size, T());
    R           val = R();
    
    Rand<T>     gen(0, 0);

    if (init_ == 0)
      view = gen.randu(size);
    else if (init_ == 1)
      view(0) = T(2);
    else if (init_ == 2)
      view(size-1) = T(2);
    else if (init_ == 3)
    {
      view(0) = 2;
      view(size-1) = T(65534);
    }
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = vsip_csl::sumval(view, R());
    t1.stop();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, R(2)));
    else
    if (init_ == 3)
      test_assert(equal(val, R(65536)));
    
    time = t1.delta();
  }

  t_sumval1_ext_base(int init) : init_(init) {}

  // member data.
  int init_;
};

template <typename T>
struct t_sumval1_ext : t_sumval1_ext_base<T,T>
{
  t_sumval1_ext(int init) : t_sumval1_ext_base<T,T>(init) {}
};

template <typename R>
struct t_sumval1_ext_help
{
  template <typename T>
  struct t_sumval1_ext : t_sumval1_ext_base<T,R>
  {
    t_sumval1_ext(int init) : t_sumval1_ext_base<T,R>(init) {}
  };
};



template <typename T>
struct t_sumval2 : Benchmark_base
{
  char const* what() { return "t_sumval_matrix32"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Matrix<T>   view(32, size, T());
    T           val = T();
    
    view(0, 1) = T(2);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = vsip::sumval(view);
    t1.stop();
    
    assert(equal(val, T(2)));
    if (!equal(val, T(2))) printf("t_sumval: ERROR\n");
    
    time = t1.delta();
  }
};



/***********************************************************************
  get/put sumval
***********************************************************************/

template <typename T>
struct t_sumval_gp : Benchmark_base
{
  char const* what() { return "t_sumval_gp"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   view(size, T());
    T           val = T();
    
    Rand<T>     gen(0, 0);

    if (init_ == 0)
      view = gen.randu(size);
    else if (init_ == 1)
      view(0) = T(2);
    else if (init_ == 2)
      view(size-1) = T(2);
    
    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      val = T();
      for (index_type i=0; i<size; ++i)
	val += view.get(i);
    }
    t1.stop();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, T(2)));
    
    time = t1.delta();
  }

  t_sumval_gp(int init) : init_(init) {}

  // member data.
  int init_;
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
  case   1: loop(t_sumval1<float>(0)); break;
  case   2: loop(t_sumval1<float>(1)); break;
  case   3: loop(t_sumval1<float>(2)); break;

  case  11: loop(t_sumval1<int>(0)); break;

  case  21: loop(t_sumval_gp<float>(0)); break;

  case  31: loop(t_sumval1<unsigned short>(0)); break;
  case  32: loop(t_sumval1_ext<unsigned short>(0)); break;
  case  33: loop(t_sumval1_ext_help<unsigned long>::t_sumval1_ext<unsigned short>(3)); break;

  case  41: loop(t_sumval1<complex<float> >(0)); break;
  case  42: loop(t_sumval1<complex<float> >(1)); break;
  case  43: loop(t_sumval1<complex<float> >(2)); break;

  case 101: loop(t_sumval2<float>()); break;
  case   0:
    std::cout
      << "sumval -- sum of the values in a view\n"
      << "   -1: vector, float, random values\n"
      << "   -2: vector, float, index is 0\n"
      << "   -3: vector, float, index is size-1\n"
      << "  -11: vector,   int, random values\n"
      << "  -21: vector, float, random values, get/put\n"
      << "  -31: vector, unsigned short, random values\n"
      << "  -32: vector, unsigned short, random values, no index result\n"
      << "  -33: vector, unsigned short, random values, no index result, unsigned long result\n"
      << "  -41: vector, complex<float>, random values\n"
      << "  -42: vector, complex<float>, index is 0\n"
      << "  -43: vector, complex<float>, index is size-1\n"
      << " -101: matrix, float\n"
      ;
  default: return 0;
  }
  return 1;
}
