/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL benchmark for 1-D Convolution 

// Originally created 2010-06-23 by Don McCoy

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;

// Set to '1' for debugging
#define VERBOSE 0


// This traits class allows the benchmark code to be written generically
// (once, that is) because it abstracts all the vsipl calls into these
// type-dependent wrappers.

template <typename T> struct conv_traits;

template <>
struct conv_traits<float>
{
  typedef vsip_vview_f input_type;
  typedef vsip_vview_f output_type;
  typedef vsip_conv1d_f conv_type;

  static void initialize()
  { vsip_init((void *)0); }
  static input_type *create_input(length_type l)
  {
    input_type *input = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(1.f, input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_valldestroy_f(v);}
  static output_type *create_output(length_type l)
  { return vsip_vcreate_f(l, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_valldestroy_f(v);}
  static conv_type *create_conv1d(length_type l, input_type* k, vsip_support_region s)
  { return vsip_conv1d_create_f(k, VSIP_NONSYM, l, 1, s, 0, VSIP_ALG_TIME); }
  static void delete_conv(conv_type *f)
  { vsip_conv1d_destroy_f(f);}
  static void convolve(conv_type *conv, input_type *input, output_type *output)
  { vsip_convolve1d_f(conv, input, output);}
  static bool valid(index_type i, output_type *output, float const &value)
  {
    vsip_scalar_f s = vsip_vget_f(output, i);
    return equal(s, value);
  }
  static void get(index_type i, output_type *output, float& value)
  {
    value = vsip_vget_f(output, i);
  }
  static void finalize()
  { vsip_finalize((void *)0); }
};

template <>
struct conv_traits<complex<float> >
{
  typedef vsip_cvview_f input_type;
  typedef vsip_cvview_f output_type;
  typedef vsip_cconv1d_f conv_type;

  static void initialize()
  { vsip_init((void *)0); }
  static input_type *create_input(length_type l)
  {
    input_type *input = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(1.f, 0.f), input);
    return input;
  }
  static void delete_input(input_type *v)
  { return vsip_cvalldestroy_f(v);}
  static output_type *create_output(length_type l)
  { return vsip_cvcreate_f(l, VSIP_MEM_NONE);}
  static void delete_output(output_type *v)
  { return vsip_cvalldestroy_f(v);}
  static conv_type *create_conv1d(length_type l, input_type* k, vsip_support_region s)
  { return vsip_cconv1d_create_f(k, VSIP_NONSYM, l, 1, s, 0, VSIP_ALG_TIME); }
  static void delete_conv(conv_type *f)
  { vsip_cconv1d_destroy_f(f);}
  static void convolve(conv_type *conv, input_type *input, output_type *output)
  { vsip_cconvolve1d_f(conv, input, output);}
  static bool valid(index_type i, output_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cvget_f(output, i);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }
  static void get(index_type i, output_type *output, std::complex<float>& value)
  {
    vsip_cscalar_f c = vsip_cvget_f(output, i);
    value.real() = c.r;
    value.imag() = c.i;
  }
  static void finalize()
  { vsip_finalize((void *)0); }
};



//  Convolution

template <support_region_type Supp,
	  typename            T>
struct t_conv1d : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_conv1d"; }
  float ops_per_point(length_type size)
  {
    float ops = coeff_size_ * output_size(size) *
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops / size;
  }

  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    vsip_support_region support;
    if      (Supp == support_full)
      support = VSIP_SUPPORT_FULL;
    else if (Supp == support_same)
      support = VSIP_SUPPORT_SAME;
    else // (Supp == support_min)
      support = VSIP_SUPPORT_MIN;

    typedef conv_traits<T> traits;
    typename traits::input_type *input = traits::create_input(size);
    typename traits::input_type *kernel = traits::create_input(coeff_size_);
    typename traits::output_type *output = traits::create_output(output_size(size));
    typename traits::conv_type *conv = traits::create_conv1d(size, kernel, support);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::convolve(conv, input, output);
    }
    t1.stop();

    T value;
    if      (Supp == support_full)
      value = T(1);
    else if (Supp == support_same)
      value = T(9);
    else // (Supp == support_min)
      value = T(16);

    index_type index = 0;  // check zeroeth element only
    if (!traits::valid(index, output, value))
    {
      std::cout << "t_conv1d: ERROR -- computed result is not valid" << std::endl;
#if VERBOSE
      std::cout << "output = ";
      for (index_type i = 0; i < 16; ++i) 
      {
        T value;
        traits::get(i, output, value);
        std::cout << value << " ";
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

  t_conv1d(length_type coeff_size)
    : coeff_size_(coeff_size)
  {
    typedef conv_traits<T> traits;
    traits::initialize();
  }

  ~t_conv1d()
  {
    typedef conv_traits<T> traits;
    traits::finalize();
  }

private:
  length_type output_size(length_type size)
  {
    length_type output_size;

    if      (Supp == support_full)
      output_size = ((size + coeff_size_ - 2)/Dec) + 1;
    else if (Supp == support_same)
      output_size = ((size-1)/Dec) + 1;
    else // (Supp == support_min)
      output_size = ((size-1)/Dec) - ((coeff_size_-1)/Dec) + 1;
    
    return output_size;
  }

  length_type coeff_size_;
};


void
defaults(Loop1P& loop)
{
  loop.start_ = 4;
  loop.user_param_ = 16;
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float> cf_type;

  switch (what)
  {
  case  1: loop(t_conv1d<support_full, float>(loop.user_param_)); break;
  case  2: loop(t_conv1d<support_same, float>(loop.user_param_)); break;
  case  3: loop(t_conv1d<support_min,  float>(loop.user_param_)); break;

  case  4: loop(t_conv1d<support_full, cf_type>(loop.user_param_)); break;
  case  5: loop(t_conv1d<support_same, cf_type>(loop.user_param_)); break;
  case  6: loop(t_conv1d<support_min,  cf_type>(loop.user_param_)); break;

  case 0:
    std::cout
      << "conv -- 1D convolution\n"
      << "   -1 -- float, support=full\n"
      << "   -2 -- float, support=same\n"
      << "   -3 -- float, support=min\n"
      << "   -4 -- complex<float>, support=full\n"
      << "   -5 -- complex<float>, support=same\n"
      << "   -6 -- complex<float>, support=min\n"
      << "\n"
      << " Parameters:\n"
      << "  -param N      -- size of coefficient vector (default 16)\n"
      << "  -start N      -- starting problem size 2^N (default 4 or 16 points)\n"
      << "  -loop_start N -- initial number of calibration loops (default 5000)\n"
      ;   

  default:
    return 0;
  }
  return 1;
}
