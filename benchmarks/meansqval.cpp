//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for meansqval reductions.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/math.hpp>
#include <vsip/random.hpp>
#include <vsip/core/reductions/functors.hpp>

#include <vsip_csl/diagnostics.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"

#include "benchmarks.hpp"

using namespace vsip;
using namespace vsip_csl;

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
    
    vsip_csl::profile::Timer t1;
    
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
    
    vsip_csl::profile::Timer t1;
    
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
struct t_meansqval_matrix : Benchmark_base
{
  char const* what() { return "t_meansqval_matrix"; }
  int ops_per_point(length_type)
  { 
    return rows_ * (vsip::impl::Ops_info<T>::add + vsip::impl::Ops_info<T>::mul);
  }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Matrix<T>   view(rows_, size, T());
    T           val = T();
    T           itm = T(rows_ * size);
    T           ans = itm * itm / T(rows_ * size);
    
    view(0, 1) = itm;
    
    vsip_csl::profile::Timer t1;
    
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

  void diag()
  {
    // NOTE: size may influence runtime dispatch behavior -- i.e. sizes
    // that are too small may not be handled by the Cell backend.
    length_type const rows = 32;
    length_type const cols = 8192;

    Matrix<T>   A(rows, cols, T());
    typedef typename Matrix<T>::block_type block_type;
    typedef typename vsip::impl::Mean_magsq_value<T>::result_type result_type;
    typedef typename dispatcher::op::reduce<vsip::impl::Mean_magsq_value> op_type;
    result_type r;

    dispatch_diagnostics<
        op_type, void,
        result_type&,
        block_type const&,
        row2_type,
        vsip::impl::Int_type<2> >
      (r, A.block(), row2_type(), vsip::impl::Int_type<2>());
  }


  t_meansqval_matrix(length_type rows)
    : rows_(rows)
  {}

private:
  length_type rows_;
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
    
    vsip_csl::profile::Timer t1;
    
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

private:
  // member data.
  int init_;
};



void
defaults(Loop1P& loop)
{
  loop.param_["rows"] = "32";
}



int
test(Loop1P& loop, int what)
{
  length_type rows = atoi(loop.param_["rows"].c_str());

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

  case 101: loop(t_meansqval_matrix<float>(rows)); break;
  case 102: loop(t_meansqval_matrix<std::complex<float> >(rows)); break;

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
      << " -102: matrix, complex float\n"
      ;
  default: return 0;
  }
  return 1;
}
