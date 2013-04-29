//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;

template <typename T> struct fir_traits;

template <>
struct fir_traits<float>
{
  typedef vsip_vview_f view_type;
  typedef vsip_fir_f fir_type;

  static view_type *create_coeff(length_type l)
  {
    view_type *coeff = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(0.f, coeff);
    vsip_vput_f(coeff, 0, 1.f);
    vsip_vput_f(coeff, 1, 2.f);
    return coeff;
  }
  static view_type *create_input(fir_type *f)
  {
    vsip_fir_attr_f attr;
    vsip_fir_getattr_f(f, &attr);
    return vsip_vcreate_f(attr.in_len, VSIP_MEM_NONE);
  }
  static view_type *create_output(fir_type *f)
  {
    vsip_fir_attr_f attr;
    vsip_fir_getattr_f(f, &attr);
    return vsip_vcreate_f(attr.out_len, VSIP_MEM_NONE);
  }
  static void delete_view(view_type *v)
  { return vsip_valldestroy_f(v);}
  static fir_type *create_fir(view_type *coeff, length_type s, length_type d, vsip_obj_state save)
  { return vsip_fir_create_f(coeff, VSIP_NONSYM, s, d, save, 0, VSIP_ALG_TIME);}
  static void delete_fir(fir_type *f)
  { vsip_fir_destroy_f(f);}
  static void fir(fir_type *fir, view_type *input, view_type *output)
  { vsip_firflt_f(fir, input, output);}
};

template <>
struct fir_traits<std::complex<float> >
{
  typedef vsip_cvview_f view_type;
  typedef vsip_cfir_f fir_type;

  static view_type *create_coeff(length_type l)
  {
    view_type *coeff = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(0.f, 0.f), coeff);
    vsip_cvput_f(coeff, 0, vsip_cmplx_f(1.f, 0.f));
    vsip_cvput_f(coeff, 1, vsip_cmplx_f(2.f, 0.f));
    return coeff;
  }
  static view_type *create_input(fir_type *f)
  {
    vsip_cfir_attr_f attr;
    vsip_cfir_getattr_f(f, &attr);
    return vsip_cvcreate_f(attr.in_len, VSIP_MEM_NONE);
  }
  static view_type *create_output(fir_type *f)
  {
    vsip_cfir_attr_f attr;
    vsip_cfir_getattr_f(f, &attr);
    return vsip_cvcreate_f(attr.out_len, VSIP_MEM_NONE);
  }
  static void delete_view(view_type *v)
  { return vsip_cvalldestroy_f(v);}
  static fir_type *create_fir(view_type *coeff, length_type s, length_type d, vsip_obj_state save)
  { return vsip_cfir_create_f(coeff, VSIP_NONSYM, s, d, save, 0, VSIP_ALG_TIME);}
  static void delete_fir(fir_type *f)
  { vsip_cfir_destroy_f(f);}
  static void fir(fir_type *fir, view_type *input, view_type *output)
  { vsip_cfirflt_f(fir, input, output);}
};


template <obj_state      Save,
	  typename       T>
struct t_fir1 : Benchmark_base
{

  char const *what() { return "t_fir1"; }

  float ops_per_point(length_type size)
  {
    float ops = (coeff_size_ * size / dec_) *
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops / size;
  }

  int riob_per_point(length_type) { return 2 * this->coeff_size_ * sizeof(T);}
  int wiob_per_point(length_type) { return this->coeff_size_ * sizeof(T);}
  int mem_per_point(length_type) { return 2 * sizeof(T);}

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef fir_traits<T> traits;
    typename traits::view_type *coeff = traits::create_coeff(size);
    typename traits::fir_type *fir = traits::create_fir(coeff, size, dec_,
                                                        Save == state_save ? VSIP_STATE_SAVE : VSIP_STATE_NO_SAVE);
    typename traits::view_type *input = traits::create_input(fir);
    typename traits::view_type *output = traits::create_output(fir);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::fir(fir, input, output);
    }
    t1.stop();
    
    time = t1.delta();

    traits::delete_view(output);
    traits::delete_view(input);
    traits::delete_fir(fir);
    traits::delete_view(coeff);
  }

  t_fir1(length_type coeff_size, length_type dec)
    : coeff_size_(coeff_size)
    , dec_       (dec)
    {}

  length_type coeff_size_;
  length_type dec_;
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 1000;
  loop.start_ = 4;

  loop.param_["k"] = "16"; // Kernel size
  loop.param_["d"] = "1";  // Decimation
}

//  Non-symmetric, non-continuous, where kernel size and decimation 
//  are parameters and input size is swept. 
//    Float and complex<float> value types.
//
//  Non-symmetric, continuous, where kernel size and decimation 
//  are parameters and input size is swept.
//    Float and complex<float> value types.

int
test(Loop1P& loop, int what)
{
  length_type k = atoi(loop.param_["k"].c_str());
  length_type d = atoi(loop.param_["d"].c_str());

  typedef float               SX;
  typedef std::complex<float> CX;

  switch (what)
  {
    case  1: loop(t_fir1<state_no_save, SX>(k, d)); break;
    case  2: loop(t_fir1<state_no_save, CX>(k, d)); break;

    case 11: loop(t_fir1<state_save,    SX>(k, d)); break;
    case 12: loop(t_fir1<state_save,    CX>(k, d)); break;

    case 0:
      std::cout
        << "fir -- FIR signal processing object benchmark\n"
        << "   -1 -- No state save, float\n"
        << "   -2 -- No state save, complex<float>\n"
        << "  -11 -- State save, float\n"
        << "  -12 -- State save, complex<float>\n"
        << "\n"
        << "Parameters\n"
        << "  -param:k <size>  Kernel size (default 16)\n"
        << "  -param:d <size>  Decimation (default 1)\n";

    default: return 0;
  }
  return 1;
}
