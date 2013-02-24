/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/meansqval.cpp
    @author  Mike LeBlanc
    @date    2009-05-02
    @brief   VSIPL++ Library: Benchmark for meansqval reductions.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/math.hpp>
#include <vsip/random.hpp>
#include <vsip/opt/profile.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

#include "benchmarks.hpp"

using namespace vsip;
using namespace vsip_csl;



/***********************************************************************
  VSIPL++ meansqval
***********************************************************************/

template <typename T>
struct t_meansqval1 : Benchmark_base
{
  char const* what() { return "t_meansqval_vector"; }
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
      view(0) = T(size);
    else if (init_ == 2)
      view(size-1) = T(size);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = vsip::meansqval(view);
    t1.stop();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, T(size)));
    
    time = t1.delta();
  }

  t_meansqval1(int init) : init_(init) {}

  // member data.
  int init_;
};

template <typename T, typename R>
struct t_meansqval1_ext_base : Benchmark_base
{
  char const* what() { return "t_meansqval_ext_vector"; }
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
      view(0) = T(size);
    else if (init_ == 2)
      view(size-1) = T(size);
    else if (init_ == 3)
    {
      view(0) = 2;
      view(size-1) = T(256);
    }
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = vsip_csl::meansqval(view, R());
    t1.stop();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, R(size)));
    else
    if (init_ == 3)
      test_assert(equal(val, R((2*2+256*256)/size)));
    
    time = t1.delta();
  }

  t_meansqval1_ext_base(int init) : init_(init) {}

  // member data.
  int init_;
};

template <typename T>
struct t_meansqval1_ext : t_meansqval1_ext_base<T,T>
{
  t_meansqval1_ext(int init) : t_meansqval1_ext_base<T,T>(init) {}
};

template <typename R>
struct t_meansqval1_ext_help
{
  template <typename T>
  struct t_meansqval1_ext : t_meansqval1_ext_base<T,R>
  {
    t_meansqval1_ext(int init) : t_meansqval1_ext_base<T,R>(init) {}
  };
};



template <typename T>
struct t_meansqval2 : Benchmark_base
{
  char const* what() { return "t_meansqval_matrix32"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Matrix<T>   view(32, size, T());
    T           val = T();
    T           itm = T(4*32*size);
    T           ans = itm*itm/T(size);
    
    view(0, 1) = itm;
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = vsip::meansqval(view);
    t1.stop();
    
    if (!equal(val, ans))
      std::cout << "t_meansqval: ERROR, val=" << val << ", size=" << size
	<< ", ans=" << ans
	<< ", itm=" << itm
	<< "\n";
    test_assert(equal(val, ans));
    
    time = t1.delta();
  }
};



/***********************************************************************
  get/put meansqval
***********************************************************************/

template <typename T>
struct t_meansqval_gp : Benchmark_base
{
  char const* what() { return "t_meansqval_gp"; }
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
      view(0) = T(size);
    else if (init_ == 2)
      view(size-1) = T(size);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      val = T();
      for (index_type i=0; i<size; ++i)
	val += view.get(i)*view.get(i);
      val /= size;
    }
    t1.stop();

    if (init_ == 1 || init_ == 2)
      test_assert(equal(val, T(size)));
    
    time = t1.delta();
  }

  t_meansqval_gp(int init) : init_(init) {}

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
  case   1: loop(t_meansqval1<float>(0)); break;
  case   2: loop(t_meansqval1<float>(1)); break;
  case   3: loop(t_meansqval1<float>(2)); break;

  case  11: loop(t_meansqval1<int>(0)); break;

  case  21: loop(t_meansqval_gp<float>(0)); break;

  case  31: loop(t_meansqval1<unsigned short>(0)); break;
  case  32: loop(t_meansqval1_ext<unsigned short>(0)); break;
  case  33: loop(t_meansqval1_ext_help<unsigned long>::t_meansqval1_ext<unsigned short>(3)); break;

  case 101: loop(t_meansqval2<float>()); break;

  case   0:
    std::cout
      << "meansqval -- mean of the squared values in a view\n"
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
