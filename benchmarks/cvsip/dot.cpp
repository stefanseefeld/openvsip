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


template <typename T> struct dot_traits;

template <>
struct dot_traits<float >
{
  typedef vsip_vview_f vector_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, 
    float const& value)
  {
    vector_type *view = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(value, view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_valldestroy_f(v); }

  static void get(index_type i, vector_type* output, float& value)
  {
    value = vsip_vget_f(output, i);
  }
  static void put(index_type i, vector_type* input, float value)
  {
    vsip_vput_f(input, i, value);
  }

  static float dot(vector_type const *a, vector_type const *b)
  { 
    return vsip_vdot_f(a, b); 
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};


template <typename T> struct dot_traits;
template <>
struct dot_traits<std::complex<float> >
{
  typedef vsip_cvview_f vector_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, 
    std::complex<float> const& value)
  {
    vector_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_cvalldestroy_f(v); }

  static void get(index_type i, vector_type* output, std::complex<float>& value)
  {
    vsip_cscalar_f cval;
    cval = vsip_cvget_f(output, i);
    value.real() = cval.r;
    value.imag() = cval.i;
  }
  static void put(index_type i, vector_type* input, std::complex<float> value)
  {
    vsip_cscalar_f cval;
    cval.r = value.real();
    cval.i = value.imag();
    vsip_cvput_f(input, i, cval);
  }

  static std::complex<float> dot(vector_type const *a, vector_type const *b)
  { 
    vsip_cscalar_f c = vsip_cvdot_f(a, b); 
    std::complex<float> value(c.r, c.i);
    return value;
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};


// Dot-product benchmark class.

template <typename T>
struct t_dot1 : Benchmark_base
{
  typedef dot_traits<T> traits;

  char const* what() { return "t_dot1"; }
  float ops_per_point(length_type)
  {
    float ops = (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);
    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type) { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typename traits::vector_type *A = traits::create_vector(size, T(3));
    typename traits::vector_type *B = traits::create_vector(size, T(4));
    T r = T();

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      r = traits::dot(A, B);
    }
      
    t1.stop();

    if (r != T(3)*T(4)*T(size))
      std::cout << "t_dot1<T>: ERROR" << std::endl;
    
    time = t1.delta();
  }

  t_dot1() {}
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_dot1<float>()); break;
  case  2: loop(t_dot1<complex<float> >()); break;

  case  0:
    std::cout
      << "dot -- dot product\n"
      << "  -1 -- float\n"
      << "  -2 -- complex<float>\n"
      << "\n"
      << " Parameters:\n"
      << "  -start N      -- starting problem size 2^N (default 4 or 16 points)\n"
      << "  -loop_start N -- initial number of calibration loops (default 5000)\n"
      ;

  default: return 0;
  }
  return 1;
}
