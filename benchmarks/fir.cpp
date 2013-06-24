//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for FIR filter.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/selgen.hpp>
#include "benchmark.hpp"

using namespace vsip;

// Sweep FIR block size, processing single block

template <obj_state      Save,
	  typename       T>
struct t_fir1 : Benchmark_base
{

  char const* what() { return "t_fir1"; }

  float ops_per_point(length_type size)
  {
    float ops = (coeff_size_ * size / dec_) *
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops / size;
  }

  int riob_per_point(length_type)
    { return 2 * this->coeff_size_ * sizeof(T); }

  int wiob_per_point(length_type)
    { return this->coeff_size_ * sizeof(T); }

  int mem_per_point(length_type)
    { return 2 * sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Fir<T,nonsym,Save> fir_type;

    Vector<T>   coeff(coeff_size_, T());
    coeff(0) = T(0);
    coeff(1) = T(1);

    fir_type fir(coeff, size, dec_);

    Vector<T>   in (size, T());
    Vector<T>   out(fir.output_size());

    in = ramp<T>(0, 1, size);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      fir(in, out);
    time = t1.elapsed();

    if (Save == state_save)
    {
      Domain<1> d_i(dec_-1, dec_, (size-1)/dec_);
      Domain<1> d_o(1, 1, (size-1)/dec_);
      float error = test::diff(in(d_i), out(d_o));
      if (error > -100)
	for (index_type i=0; i<d_i.size(); ++i)
	  std::cout << i << ": " << in(d_i.impl_nth(i)) << ", "
		    << out(d_o.impl_nth(i)) << std::endl;
      test_assert(error < -100);
    }
  }

  t_fir1(length_type coeff_size, length_type dec)
    : coeff_size_(coeff_size)
    , dec_       (dec)
    {}

  void diag()
  {
    typedef Fir<T,nonsym,Save> fir_type;

    Vector<T>   coeff(coeff_size_, T());
    coeff(0) = T(1);
    coeff(1) = T(2);

    length_type size = 1024;

    fir_type fir(coeff, size, dec_);
    // TBD
  }

  length_type coeff_size_;
  length_type dec_;
};



// Sweep FIR block size, processing fixed input size

template <obj_state      Save,
	  typename       T>
struct t_fir2 : Benchmark_base
{

  char const* what() { return "t_fir2"; }

  float ops_per_point(length_type size)
  {
    float ops = (coeff_size_ * total_size_) *
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops / size;
  }

  float riob_per_point(length_type size)
    { return 2 * total_size_ * sizeof(T) / size; }

  float wiob_per_point(length_type size)
    { return total_size_ * sizeof(T) / size; }

  int mem_per_point(length_type size)
    { return 2 * sizeof(T) * total_size_ / size; }

  void operator()(length_type block_size, length_type loop, float& time)
  {
    typedef Fir<T,nonsym,Save> fir_type;

    Vector<T>   coeff(coeff_size_, T());
    coeff(0) = T(1);
    coeff(1) = T(2);

    fir_type fir(coeff, block_size, 1);

    Vector<T>   in (total_size_, T());
    Vector<T>   out(total_size_);

    timer t1;
    for (index_type l=0; l<loop; ++l)
    {
      for (index_type pos=0; pos<total_size_; pos += block_size)
	fir(in (Domain<1>(pos, 0, block_size)),
	    out(Domain<1>(pos, 0, block_size)));
    }
    time = t1.elapsed();
  }

  t_fir2(length_type coeff_size, length_type total_size)
    : coeff_size_(coeff_size)
    , total_size_(total_size)
    {}

  void diag()
  {
    typedef Fir<T,nonsym,Save> fir_type;

    Vector<T>   coeff(coeff_size_, T());
    coeff(0) = T(1);
    coeff(1) = T(2);

    length_type block_size = 1024;

    fir_type fir(coeff, block_size, 1);
    // TBD
  }

  length_type coeff_size_;
  length_type total_size_;
};






void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 1;
  loop.start_ = 4;
  loop.cal_   = 4;

  loop.param_["k"]    = "16"; // Kernel size
  loop.param_["d"]    = "1";  // Decimation
  loop.param_["size"] = "0";  // Size
}

//  Non-symmetric, non-continuous, where kernel size and decimation 
//  are parameters and input size is swept. 
//    Float and complex<float> value types.
//
//  Non-symmetric, continuous, where kernel size and decimation 
//  are parameters and input size is swept.
//    Float and complex<float> value types.

int
benchmark(Loop1P& loop, int what)
{
  length_type k = atoi(loop.param_["k"].c_str());
  length_type d = atoi(loop.param_["d"].c_str());
  length_type size = atoi(loop.param_["size"].c_str());

  if (size == 0)
    size = 1 << loop.stop_;

  typedef float               SX;
  typedef std::complex<float> CX;

  switch (what)
  {
  case  1: loop(t_fir1<state_no_save, SX>(k, d)); break;
  case  2: loop(t_fir1<state_no_save, CX>(k, d)); break;

  case 11: loop(t_fir1<state_save,    SX>(k, d)); break;
  case 12: loop(t_fir1<state_save,    CX>(k, d)); break;

  case 21: loop(t_fir2<state_no_save, SX>(k, size)); break;
  case 22: loop(t_fir2<state_no_save, CX>(k, size)); break;
  case 31: loop(t_fir2<state_save,    SX>(k, size)); break;
  case 32: loop(t_fir2<state_save,    CX>(k, size)); break;

  case 0:
    std::cout
      << "fir -- FIR signal processing object benchmark\n"
      << " Sweep block size == problem size\n"
      << "   -1 -- No state save, float\n"
      << "   -2 -- No state save, complex<float>\n"
      << "  -11 -- State save,    float\n"
      << "  -12 -- State save,    complex<float>\n"
      << "\n"
      << "Parameters for cases 1, 2, 11, 12\n"
      << "  -p:k <size>  Kernel size (default 16)\n"
      << "  -p:d <size>  Decimation (default 1)\n"
      << "\n"
      << " Sweep block size, fixed problem size\n"
      << "  -21 -- No state save, float\n"
      << "  -22 -- No state save, complex<float>\n"
      << "  -31 -- State save,    float\n"
      << "  -32 -- State save,    complex<float>\n"
      << "\n"
      << "Parameters for cases 22, 32\n"
      << "  -p:k <size>  Kernel size (default 16)\n"
      << "  -p:size <size> Problem size (default 1)\n"
      ;

  default: return 0;
  }
  return 1;
}
