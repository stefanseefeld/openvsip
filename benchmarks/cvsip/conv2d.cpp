//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;

// Set to '1' for debugging
#define VERBOSE 0


// This traits class allows the benchmark code to be written generically
// (once, that is) because it abstracts all the vsipl calls into these
// type-dependent wrappers.

template <typename T> struct conv2d_traits;

template <>
struct conv2d_traits<float>
{
  typedef vsip_mview_f input_type;
  typedef vsip_mview_f output_type;
  typedef vsip_conv2d_f conv_type;

  static void initialize()
  { vsip_init((void *)0); }
  static input_type *create_input(length_type r, length_type c)
  {
    input_type *input = vsip_mcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_mfill_f(1.f, input);
    return input;
  }
  static void delete_input(input_type *m)
  { return vsip_malldestroy_f(m);}
  static output_type *create_output(length_type r, length_type c)
  { return vsip_mcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);}
  static void delete_output(output_type *m)
  { return vsip_malldestroy_f(m);}
  static conv_type *create_conv2d(length_type r, length_type c, input_type* k, vsip_support_region s)
  { return vsip_conv2d_create_f(k, VSIP_NONSYM, r, c, 1, s, 0, VSIP_ALG_TIME); }
  static void delete_conv(conv_type *f)
  { vsip_conv2d_destroy_f(f);}
  static void convolve(conv_type *conv, input_type *input, output_type *output)
  { vsip_convolve2d_f(conv, input, output);}
  static bool valid(index_type i, index_type j, output_type *output, float const &value)
  {
    vsip_scalar_f s = vsip_mget_f(output, i, j);
    return equal(s, value);
  }
  static void get(index_type i, index_type j, output_type *output, float& value)
  {
    value = vsip_mget_f(output, i, j);
  }
  static void finalize()
  { vsip_finalize((void *)0); }
};

template <>
struct conv2d_traits<complex<float> >
{
  typedef vsip_cmview_f input_type;
  typedef vsip_cmview_f output_type;
  typedef vsip_cconv2d_f conv_type;

  static void initialize()
  { vsip_init((void *)0); }
  static input_type *create_input(length_type r, length_type c)
  {
    input_type *input = vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_cmfill_f(vsip_cmplx_f(1.f, 0.f), input);
    return input;
  }
  static void delete_input(input_type *m)
  { return vsip_cmalldestroy_f(m);}
  static output_type *create_output(length_type r, length_type c)
  { return vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);}
  static void delete_output(output_type *m)
  { return vsip_cmalldestroy_f(m);}
  static conv_type *create_conv2d(length_type r, length_type c, input_type* k, vsip_support_region s)
  { return vsip_cconv2d_create_f(k, VSIP_NONSYM, r, c, 1, s, 0, VSIP_ALG_TIME); }
  static void delete_conv(conv_type *f)
  { vsip_cconv2d_destroy_f(f);}
  static void convolve(conv_type *conv, input_type *input, output_type *output)
  { vsip_cconvolve2d_f(conv, input, output);}
  static bool valid(index_type i, index_type j, output_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cmget_f(output, i, j);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }
  static void get(index_type i, index_type j, output_type *output, std::complex<float>& value)
  {
    vsip_cscalar_f c = vsip_cmget_f(output, i, j);
    value.real() = c.r;
    value.imag() = c.i;
  }
  static void finalize()
  { vsip_finalize((void *)0); }
};



//  Convolution

template <support_region_type Supp,
	  typename            T>
struct t_conv2d : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_conv2d"; }

  float ops_per_point(length_type cols)
  {
    length_type o_rows, o_cols;

    output_size(rows_, cols, o_rows, o_cols);

    float ops = m_ * n_ * o_rows * o_cols *
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops / cols;
  }

  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    length_type o_rows, o_cols;
    output_size(rows_, cols, o_rows, o_cols);

    vsip_support_region support;
    if      (Supp == support_full)
      support = VSIP_SUPPORT_FULL;
    else if (Supp == support_same)
      support = VSIP_SUPPORT_SAME;
    else // (Supp == support_min)
      support = VSIP_SUPPORT_MIN;

    typedef conv2d_traits<T> traits;
    typename traits::input_type *input = traits::create_input(rows_, cols);
    typename traits::input_type *kernel = traits::create_input(m_, n_);
    typename traits::output_type *output = traits::create_output(o_rows, o_cols);
    typename traits::conv_type *conv = traits::create_conv2d(rows_, cols, kernel, support);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      traits::convolve(conv, input, output);
    t1.stop();

    T value;
    if      (Supp == support_full)
      value = T(1);
    else if (Supp == support_same)
      value = T(4);
    else // (Supp == support_min)
      value = T(9);

    index_type index = 0;  // check zeroeth element only
    if (!traits::valid(index, index, output, value))
    {
      std::cout << "t_conv2d: ERROR -- computed result is not valid" << std::endl;
#if VERBOSE
      std::cout << "output = ";
      for (index_type i = 0; i < 16; ++i) 
      {
        for (index_type j = 0; j < 4; ++j) 
        {
          T value;
          traits::get(i, j, output, value);
          std::cout << value << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
#endif
    }
    
    time = t1.delta();

    traits::delete_conv(conv);
    traits::delete_output(output);
    traits::delete_input(kernel);
    traits::delete_input(input);
  }

  t_conv2d(length_type rows, length_type m, length_type n)
    : rows_(rows), m_(m), n_(n)
  {
    typedef conv2d_traits<T> traits;
    traits::initialize();
  }

  ~t_conv2d()
  {
    typedef conv2d_traits<T> traits;
    traits::finalize();
  }

private:
  void output_size(
    length_type  rows,   length_type  cols,
    length_type& o_rows, length_type& o_cols)
  {
    length_type const rdec = 1;
    length_type const cdec = 1;

    if      (Supp == support_full)
    {
      o_rows = ((rows + m_ - 2)/rdec) + 1;
      o_cols = ((cols + n_ - 2)/cdec) + 1;
    }
    else if (Supp == support_same)
    {
      o_rows = ((rows - 1)/rdec) + 1;
      o_cols = ((cols - 1)/cdec) + 1;
    }
    else /* (Supp == support_min) */
    {
      o_rows = ((rows-1)/rdec) - ((m_-1)/rdec) + 1;
      o_cols = ((cols-1)/cdec) - ((n_-1)/rdec) + 1;
    }
  }
  
  length_type rows_;
  length_type m_;   // coeff rows
  length_type n_;   // coeff cols
};


void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;

  loop.param_["rows"] = "16";
  loop.param_["mn"]   = "0";
  loop.param_["m"]    = "3";
  loop.param_["n"]    = "3";
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float> cf_type;

  length_type rows = atoi(loop.param_["rows"].c_str());
  length_type MN   = atoi(loop.param_["mn"].c_str());
  length_type M    = atoi(loop.param_["m"].c_str());
  length_type N    = atoi(loop.param_["n"].c_str());

  if (MN != 0)
    M = N = MN;

  switch (what)
  {
  case  1: loop(t_conv2d<support_full, float>(rows, M, N)); break;
  case  2: loop(t_conv2d<support_same, float>(rows, M, N)); break;
  case  3: loop(t_conv2d<support_min, float> (rows, M, N)); break;

  case  4: loop(t_conv2d<support_full, cf_type>(rows, M, N)); break;
  case  5: loop(t_conv2d<support_same, cf_type>(rows, M, N)); break;
  case  6: loop(t_conv2d<support_min,  cf_type>(rows, M, N)); break;

  case 0:
    std::cout
      << "conv2d -- 2D convolution\n"
      << "   -1 -- float, support=full\n"
      << "   -2 -- float, support=same\n"
      << "   -3 -- float, support=min\n"
      << "\n"
      << " Parameters:\n"
      << "  -p:rows     ROWS -- rows in input matrix (default 16)\n"
      << "  -p:m        M    -- rows in coefficient matrix (default 3)\n"
      << "  -p:n        N    -- columns in coefficient matrix (default 3)\n"


      << "  -p:mn       MN   -- if not zero, set both M and N to MN (default 0)\n"
      << "  -start      N    -- starting problem size 2^N (default 4, that is 16 points)\n"
      << "  -loop_start N    -- initial number of calibration loops (default 5000)\n"
      ;   

  default:
    return 0;
  }
  return 1;
}
