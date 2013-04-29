//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef BENCHMARKS_FASTCONV_HPP
#define BENCHMARKS_FASTCONV_HPP

/***********************************************************************
  Macros
***********************************************************************/

#ifdef VSIP_IMPL_SOURCERY_VPP
#  define PARALLEL_FASTCONV 1
#else
#  define PARALLEL_FASTCONV 0
#endif



/***********************************************************************
  Common definitions
***********************************************************************/

template <typename T,
	  typename ImplTag>
struct t_fastconv_base;



struct fastconv_ops : Benchmark_base
{
  float ops(vsip::length_type npulse, vsip::length_type nrange) 
  {
    float fft_ops = 5 * nrange * std::log((float)nrange) / std::log(2.f);
    float tot_ops = 2 * npulse * fft_ops + 6 * npulse * nrange;
    return tot_ops;
  }
};




/***********************************************************************
  PF driver: (P)ulse (F)ixed
***********************************************************************/

template <typename T, typename ImplTag>
struct t_fastconv_pf : public t_fastconv_base<T, ImplTag>
{
  char const* what() { return "t_fastconv_pf"; }
  float ops_per_point(vsip::length_type size)
    { return this->ops(npulse_, size) / size; }
  int riob_per_point(vsip::length_type) { return npulse_*(int)sizeof(T); }
  int wiob_per_point(vsip::length_type) { return npulse_*(int)sizeof(T); }
  int mem_per_point(vsip::length_type)  { return this->num_args*npulse_*sizeof(T); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->fastconv(npulse_, size, loop, time);
  }

  t_fastconv_pf(vsip::length_type npulse) : npulse_(npulse) {}

// Member data
  vsip::length_type npulse_;
};



/***********************************************************************
  RF driver: (R)ange cells (F)ixed
***********************************************************************/

template <typename T, typename ImplTag>
struct t_fastconv_rf : public t_fastconv_base<T, ImplTag>
{
  char const* what() { return "t_fastconv_rf"; }
  float ops_per_point(vsip::length_type size)
    { return this->ops(size, nrange_) / size; }
  int riob_per_point(vsip::length_type) { return nrange_*(int)sizeof(T); }
  int wiob_per_point(vsip::length_type) { return nrange_*(int)sizeof(T); }
  int mem_per_point(vsip::length_type)  { return this->num_args*nrange_*sizeof(T); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->fastconv(size, nrange_, loop, time);
  }

  t_fastconv_rf(vsip::length_type nrange) : nrange_(nrange) {}

// Member data
  vsip::length_type nrange_;
};

#endif // BENCHMARKS_FASTCONV_HPP
