/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for VSIPL vector multiply-add.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;


template <typename T> struct vma_traits;

template <>
struct vma_traits<float>
{
  typedef vsip_vview_f vector_type;
  typedef float scalar_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, float const& value)
  {
    vector_type *view = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(value, view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_valldestroy_f(v); }

  static void put(index_type i, vector_type* input, float value)
  {
    vsip_vput_f(input, i, value);
  }

  static void vma(
    vector_type const *A, vector_type const *B, vector_type const *C, 
    vector_type const *R)
  { 
    vsip_vma_f(A, B, C, R);
  }

  static void vsma(
    vector_type const *A, scalar_type beta, vector_type const *C, 
    vector_type const *R)
  { 
    vsip_vsma_f(A, beta, C, R);
  }

  static void vsmsa(
    vector_type const *A, scalar_type beta, scalar_type gamma, 
    vector_type const *R)
  { 
    vsip_vsmsa_f(A, beta, gamma, R);
  }

  static bool equal(vector_type const *A, scalar_type value, index_type i)
  { 
    return ::equal(vsip_vget_f(A, i), value);
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};

template <>
struct vma_traits<std::complex<float> >
{
  typedef vsip_cvview_f vector_type;
  typedef std::complex<float> scalar_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, std::complex<float> const& value)
  {
    vector_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_cvalldestroy_f(v); }

  static void put(index_type i, vector_type* input, std::complex<float> value)
  {
    vsip_cvput_f(input, i, vsip_cmplx_f(value.real(), value.imag()));
  }

  static void vma(
    vector_type const *A, vector_type const *B, vector_type const *C, 
    vector_type const *R)
  { 
    vsip_cvma_f(A, B, C, R);
  }
  static void vsma(
    vector_type const *A, scalar_type beta, vector_type const *C, 
    vector_type const *R)
  { 
    vsip_cvsma_f(A, vsip_cmplx_f(beta.real(), beta.imag()), C, R);
  }
  static void vsmsa(
    vector_type const *A, scalar_type beta, scalar_type gamma, 
    vector_type const *R)
  { 
    vsip_cvsmsa_f(A, 
      vsip_cmplx_f(beta.real(), beta.imag()),
      vsip_cmplx_f(gamma.real(), gamma.imag()), R);
  }

  static bool equal(vector_type const *A, scalar_type value, index_type i)
  { 
    vsip_cscalar_f c = vsip_cvget_f(A, i);
    return ::equal(c.r, value.real()) && ::equal(c.i, value.imag());
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};




/***********************************************************************
  Definitions - vector element-wise fused multiply-add
***********************************************************************/

template <typename T,
	  dimension_type DimA,
	  dimension_type DimB,
	  dimension_type DimC>
struct t_vma : Benchmark_base
{
  typedef vma_traits<T> traits;

  char const* what() { return "t_vma"; }
  int ops_per_point(length_type)
    { return vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typename traits::vector_type *A = traits::create_vector(size, T(3));
    typename traits::vector_type *B = traits::create_vector(size, T(4));
    typename traits::vector_type *C = traits::create_vector(size, T(5));
    typename traits::scalar_type beta = T(4);
    typename traits::scalar_type gamma = T(5);
    typename traits::vector_type *R = traits::create_vector(size, T());

    vsip_csl::profile::Timer t1;

    // Determine which are scalar and which are vector.  Note that the order
    // of these dimensions does not match the order in the VSIPL call.
    test_assert(DimB == 1);
    bool is_vma = (DimA == 1) && (DimC == 1);
    bool is_vsma = (DimA == 0) && (DimC == 1);
//  bool is_vsmsa = (DimA == 0) && (DimC == 0);

    t1.start();
    if (is_vma)
    {
      for (index_type l=0; l<loop; ++l)
      {
        traits::vma(A, B, C, R);
      }
    }
    else if (is_vsma)
    {
      for (index_type l=0; l<loop; ++l)
      {
        traits::vsma(A, beta, C, R);
      }
    }
    else // is_vsmsa
    {
      for (index_type l=0; l<loop; ++l)
      {
        traits::vsmsa(A, beta, gamma, R);
      }
    }
    t1.stop();
    
    // Sanity check
    if (!traits::equal(R, T(3*4+5), 0) ||
        !traits::equal(R, T(3*4+5), size/2) ||
        !traits::equal(R, T(3*4+5), size-1))
    {
      std::cout << "t_vma: ERROR" << std::endl;
    }
    
    time = t1.delta();
  }

  t_vma()
  {
    traits::initialize();
  }

  ~t_vma()
  {
    traits::finalize();
  }
};


void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  typedef float           SF;
  typedef complex<float>  CF;

  switch (what)
  {
  case   1: loop(t_vma<SF, 1, 1, 1>()); break;
  case   2: loop(t_vma<SF, 0, 1, 1>()); break;
  case   3: loop(t_vma<SF, 0, 1, 0>()); break;

  case  11: loop(t_vma<CF, 1, 1, 1>()); break;
  case  12: loop(t_vma<CF, 0, 1, 1>()); break;
  case  13: loop(t_vma<CF, 0, 1, 0>()); break;

  case 0:
    std::cout
      << "vma -- vector multiply-add\n"
      << "   -1 -- V = A * B + C [float]\n"
      << "   -2 -- V = a * B + C [float]\n"
      << "   -3 -- V = a * B + c [float]\n"
      << "  -11 -- V = A * B + C [complex]\n"
      << "  -12 -- V = a * B + C [complex]\n"
      << "  -13 -- V = a * B + c [complex]\n"
      << std::endl;
  default:
    return 0;
  }
  return 1;
}
