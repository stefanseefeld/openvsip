//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for sumsqval reductions.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "benchmark.hpp"

using namespace ovxx;

template <typename T>
struct t_sumsqval1 : Benchmark_base
{
  char const* what() { return "t_sumsqval_vector"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::add+ovxx::ops_count::traits<T>::mul; }
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
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      val = vsip::sumsqval(view);
    time = t1.elapsed();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, T(4)));
  }

  t_sumsqval1(int init) : init_(init) {}

  // member data.
  int init_;
};

template <typename T, typename R>
struct t_sumsqval1_ext_base : Benchmark_base
{
  char const* what() { return "t_sumsqval_ext_vector"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::add+ovxx::ops_count::traits<T>::mul; }
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
      view(size-1) = T(256);
    }
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      val = sumsqval(view, R());
    time = t1.elapsed();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, R(4)));
    else
    if (init_ == 3)
      test_assert(equal(val, R(65540)));
  }

  t_sumsqval1_ext_base(int init) : init_(init) {}

  // member data.
  int init_;
};

template <typename T>
struct t_sumsqval1_ext : t_sumsqval1_ext_base<T,T>
{
  t_sumsqval1_ext(int init) : t_sumsqval1_ext_base<T,T>(init) {}
};

template <typename R>
struct t_sumsqval1_ext_help
{
  template <typename T>
  struct t_sumsqval1_ext : t_sumsqval1_ext_base<T,R>
  {
    t_sumsqval1_ext(int init) : t_sumsqval1_ext_base<T,R>(init) {}
  };
};



template <typename T>
struct t_sumsqval2 : Benchmark_base
{
  char const* what() { return "t_sumsqval_matrix32"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::add+ovxx::ops_count::traits<T>::mul; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Matrix<T>   view(32, size, T());
    T           val = T();
    
    view(0, 1) = T(2);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      val = vsip::sumsqval(view);
    time = t1.elapsed();
    
    test_assert(equal(val, T(4)));
  }
};



/***********************************************************************
  get/put sumsqval
***********************************************************************/

template <typename T>
struct t_sumsqval_gp : Benchmark_base
{
  char const* what() { return "t_sumsqval_gp"; }
  int ops_per_point(length_type)  { return ovxx::ops_count::traits<T>::add+ovxx::ops_count::traits<T>::mul; }
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
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      val = T();
      for (index_type i=0; i<size; ++i)
	val += view.get(i)*view.get(i);
    }
    time = t1.elapsed();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, T(4)));
  }

  t_sumsqval_gp(int init) : init_(init) {}

  // member data.
  int init_;
};



void
defaults(Loop1P&)
{
}



int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case   1: loop(t_sumsqval1<float>(0)); break;
  case   2: loop(t_sumsqval1<float>(1)); break;
  case   3: loop(t_sumsqval1<float>(2)); break;

  case  11: loop(t_sumsqval1<int>(0)); break;

  case  21: loop(t_sumsqval_gp<float>(0)); break;

  case  31: loop(t_sumsqval1<unsigned short>(0)); break;
    // TODO: This requires an extension API that is currently not implemented
  // case  32: loop(t_sumsqval1_ext<unsigned short>(0)); break;
  // case  33: loop(t_sumsqval1_ext_help<unsigned long>::t_sumsqval1_ext<unsigned short>(3)); break;

  case 101: loop(t_sumsqval2<float>()); break;

  case   0:
    std::cout
      << "sumsqval -- sum of the squared values in a view\n"
      << "   -1: vector, float, random values\n"
      << "   -2: vector, float, index is 0\n"
      << "   -3: vector, float, index is size-1\n"
      << "  -11: vector,   int, random values\n"
      << "  -21: vector, float, random values, get/put\n"
      << "  -31: vector, unsigned short, random values\n"
      << "  -32: vector, unsigned short, random values, no index result\n"
      << "  -33: vector, unsigned short, random values, no index result, unsigned long result\n"
      << " -101: matrix, float\n"
      ;
  default: return 0;
  }
  return 1;
}
