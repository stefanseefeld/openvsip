/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for vector multiply.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;

template <typename T1, typename T2 = T1> struct vmul_traits;

template <>
struct vmul_traits<float>
{
  typedef vsip_vview_f view1_type;
  typedef vsip_vview_f view2_type;

  static int ops_per_point()  { return vsip::impl::Ops_info<float>::mul;}

  static view1_type *create_view1(length_type l)
  {
    view1_type *view = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(0.f, view);
    return view;
  }
  static view2_type *create_view2(length_type l)
  {
    view2_type *view = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(0.f, view);
    return view;
  }
  static void delete_view(view1_type *v) { vsip_valldestroy_f(v);}

  static void put(view1_type *A, index_type i, float value)
  { vsip_vput_f(A, i, value);}
  static void randu(view1_type *A, index_type i, vsip_randstate *rnd)
  { vsip_vput_f(A, i, vsip_randu_f(rnd));}

  static void vmul(view1_type const *A, view1_type const *B, view1_type *C)
  { vsip_vmul_f(A, B, C);}

  static bool equal(view1_type const *A, view1_type const *B, view1_type const *C,
		    index_type i)
  { return ::equal(vsip_vget_f(A, i) * vsip_vget_f(B, i), vsip_vget_f(C, i));}
};

template <>
struct vmul_traits<double>
{
  typedef vsip_vview_d view1_type;
  typedef vsip_vview_d view2_type;

  static int ops_per_point()  { return vsip::impl::Ops_info<double>::mul;}

  static view1_type *create_view1(length_type l)
  {
    view1_type *view = vsip_vcreate_d(l, VSIP_MEM_NONE);
    vsip_vfill_d(0.f, view);
    return view;
  }
  static view2_type *create_view2(length_type l)
  {
    view2_type *view = vsip_vcreate_d(l, VSIP_MEM_NONE);
    vsip_vfill_d(0.f, view);
    return view;
  }
  static void delete_view(view1_type *v) { vsip_valldestroy_d(v);}

  static void put(view1_type *A, index_type i, float value)
  { vsip_vput_d(A, i, value);}
  static void randu(view1_type *A, index_type i, vsip_randstate *rnd)
  { vsip_vput_d(A, i, vsip_randu_d(rnd));}

  static void vmul(view1_type const *A, view1_type const *B, view1_type *C)
  { vsip_vmul_d(A, B, C);}

  static bool equal(view1_type const *A, view1_type const *B, view1_type const *C,
		    index_type i)
  { return ::equal(vsip_vget_d(A, i) * vsip_vget_d(B, i), vsip_vget_d(C, i));}
};

template <>
struct vmul_traits<std::complex<float> >
{
  typedef vsip_cvview_f view1_type;
  typedef vsip_cvview_f view2_type;

  static int ops_per_point()
  { return vsip::impl::Ops_info<std::complex<float> >::mul;}

  static view1_type *create_view1(length_type l)
  {
    view1_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(0.f, 0.f), view);
    return view;
  }
  static view2_type *create_view2(length_type l)
  {
    view1_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(0.f, 0.f), view);
    return view;
  }
  static void delete_view(view1_type *v) { vsip_cvalldestroy_f(v);}

  static void put(view1_type *A, index_type i, std::complex<float> const &value)
  { vsip_cvput_f(A, i, vsip_cmplx_f(value.real(), value.imag()));}
  static void randu(view1_type *A, index_type i, vsip_randstate *rnd)
  { vsip_cvput_f(A, i, vsip_crandu_f(rnd));}

  static void vmul(view1_type const *A, view1_type const *B, view1_type *C)
  { vsip_cvmul_f(A, B, C);}

  static bool equal(view1_type const *A, view1_type const *B, view1_type const *C,
		    index_type i)
  {
    vsip_cscalar_f lhs = vsip_cmul_f(vsip_cvget_f(A, i), vsip_cvget_f(B, i));
    vsip_cscalar_f rhs = vsip_cvget_f(C, i);
    return ::equal(lhs.r, rhs.r) && ::equal(lhs.i, rhs.i);
  }
};

template <>
struct vmul_traits<std::complex<double> >
{
  typedef vsip_cvview_d view1_type;
  typedef vsip_cvview_d view2_type;

  static int ops_per_point()
  { return vsip::impl::Ops_info<std::complex<double> >::mul;}

  static view1_type *create_view1(length_type l)
  {
    view1_type *view = vsip_cvcreate_d(l, VSIP_MEM_NONE);
    vsip_cvfill_d(vsip_cmplx_d(0.f, 0.), view);
    return view;
  }
  static view2_type *create_view2(length_type l)
  {
    view1_type *view = vsip_cvcreate_d(l, VSIP_MEM_NONE);
    vsip_cvfill_d(vsip_cmplx_d(0.f, 0.), view);
    return view;
  }
  static void delete_view(view1_type *v) { vsip_cvalldestroy_d(v);}

  static void put(view1_type *A, index_type i, std::complex<double> const &value)
  { vsip_cvput_d(A, i, vsip_cmplx_d(value.real(), value.imag()));}
  static void randu(view1_type *A, index_type i, vsip_randstate *rnd)
  { vsip_cvput_d(A, i, vsip_crandu_d(rnd));}

  static void vmul(view1_type const *A, view1_type const *B, view1_type *C)
  { vsip_cvmul_d(A, B, C);}

  static bool equal(view1_type const *A, view1_type const *B, view1_type const *C,
		    index_type i)
  {
    vsip_cscalar_d lhs = vsip_cmul_d(vsip_cvget_d(A, i), vsip_cvget_d(B, i));
    vsip_cscalar_d rhs = vsip_cvget_d(C, i);
    return ::equal(lhs.r, rhs.r) && ::equal(lhs.i, rhs.i);
  }
};

template <>
struct vmul_traits<std::complex<float>, float>
{
  typedef vsip_cvview_f view1_type;
  typedef vsip_vview_f view2_type;

  static int ops_per_point() { return 2 * vsip::impl::Ops_info<float>::mul;}

  static view1_type *create_view1(length_type l)
  {
    view1_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(0.f, 0.f), view);
    return view;
  }
  static view2_type *create_view2(length_type l)
  {
    view2_type *view = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(0.f, view);
    return view;
  }
  static void delete_view(view1_type *v) { vsip_cvalldestroy_f(v);}
  static void delete_view(view2_type *v) { vsip_valldestroy_f(v);}

  static void put(view1_type *A, index_type i, std::complex<float> const &value)
  { vsip_cvput_f(A, i, vsip_cmplx_f(value.real(), value.imag()));}
  static void put(view2_type *A, index_type i, float value)
  { vsip_vput_f(A, i, value);}
  static void randu(view1_type *A, index_type i, vsip_randstate *rnd)
  { vsip_cvput_f(A, i, vsip_crandu_f(rnd));}
  static void randu(view2_type *A, index_type i, vsip_randstate *rnd)
  { vsip_vput_f(A, i, vsip_randu_f(rnd));}

  static void vmul(view1_type const *A, view2_type const *B, view1_type *C)
  { vsip_rcvmul_f(B, A, C);}

  static bool equal(view1_type const *A, view2_type const *B, view1_type const *C,
		    index_type i)
  {
    vsip_cscalar_f lhs = vsip_rcmul_f(vsip_vget_f(B, i), vsip_cvget_f(A, i));
    vsip_cscalar_f rhs = vsip_cvget_f(C, i);
    return ::equal(lhs.r, rhs.r) && ::equal(lhs.i, rhs.i);
  }
};

template <>
struct vmul_traits<std::complex<double>, double>
{
  typedef vsip_cvview_d view1_type;
  typedef vsip_vview_d view2_type;

  static int ops_per_point() { return 2 * vsip::impl::Ops_info<double>::mul;}

  static view1_type *create_view1(length_type l)
  {
    view1_type *view = vsip_cvcreate_d(l, VSIP_MEM_NONE);
    vsip_cvfill_d(vsip_cmplx_d(0.f, 0.), view);
    return view;
  }
  static view2_type *create_view2(length_type l)
  {
    view2_type *view = vsip_vcreate_d(l, VSIP_MEM_NONE);
    vsip_vfill_d(0., view);
    return view;
  }
  static void delete_view(view1_type *v) { vsip_cvalldestroy_d(v);}
  static void delete_view(view2_type *v) { vsip_valldestroy_d(v);}

  static void put(view1_type *A, index_type i, std::complex<double> const &value)
  { vsip_cvput_d(A, i, vsip_cmplx_d(value.real(), value.imag()));}
  static void put(view2_type *A, index_type i, double value)
  { vsip_vput_d(A, i, value);}
  static void randu(view1_type *A, index_type i, vsip_randstate *rnd)
  { vsip_cvput_d(A, i, vsip_crandu_d(rnd));}
  static void randu(view2_type *A, index_type i, vsip_randstate *rnd)
  { vsip_vput_d(A, i, vsip_randu_d(rnd));}

  static void vmul(view1_type const *A, view2_type const *B, view1_type *C)
  { vsip_rcvmul_d(B, A, C);}

  static bool equal(view1_type const *A, view2_type const *B, view1_type const *C,
		    index_type i)
  {
    vsip_cscalar_d lhs = vsip_rcmul_d(vsip_vget_d(B, i), vsip_cvget_d(A, i));
    vsip_cscalar_d rhs = vsip_cvget_d(C, i);
    return ::equal(lhs.r, rhs.r) && ::equal(lhs.i, rhs.i);
  }
};

// Either T1 and T2 are equal, or T2 is scalar_of<T1>::type
template <typename T1, typename T2>
struct t_vmul1 : Benchmark_base
{
  typedef Dense<1, T1, row1_type> block1_type;
  typedef Dense<1, T2, row1_type> block2_type;
  typedef vmul_traits<T1, T2> traits;

  char const* what() { return "t_vmul1"; }
  int ops_per_point(length_type)  { return traits::ops_per_point();}
  int riob_per_point(length_type) { return sizeof(T1) + sizeof(T2);}
  int wiob_per_point(length_type) { return sizeof(T1);}
  int mem_per_point(length_type)  { return 2*sizeof(T1) + sizeof(T2);}

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typename traits::view1_type *A = traits::create_view1(size);
    typename traits::view2_type *B = traits::create_view2(size);
    typename traits::view1_type *C = traits::create_view1(size);
    vsip_randstate *rng = vsip_randcreate(0, 1, 1, VSIP_PRNG);

    for (index_type i = 0; i < size; ++i)
    {
      traits::randu(A, i, rng);
      traits::randu(B, i, rng);
    }
    traits::put(A, 0, 3);
    traits::put(B, 0, 4);

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l = 0; l < loop; ++l)
    {
      traits::vmul(A, B, C);
    }
    t1.stop();
    
    for (index_type i = 0; i < size; ++i)
      test_assert(traits::equal(A, B, C, i));
    
    time = t1.delta();
    
    vsip_randdestroy(rng);
    traits::delete_view(C);
    traits::delete_view(B);
    traits::delete_view(A);
  }

};

/***********************************************************************
  Definitions
***********************************************************************/

void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case   1: loop(t_vmul1<float, float>()); break;
  case   2: loop(t_vmul1<complex<float>, complex<float> >()); break;

  case   5: loop(t_vmul1<std::complex<float>, float>()); break;

  // Double-precision

  case 101: loop(t_vmul1<double, double>()); break;
  case 102: loop(t_vmul1<complex<double>, complex<double> >()); break;

  case 105: loop(t_vmul1<std::complex<double>, double>()); break;

  case 0:
    std::cout
      << "vmul -- vector multiplication\n"
      << "single-precision:\n"
      << " Vector-Vector:\n"
      << "   -1 -- Vector<        float > * Vector<        float >\n"
      << "   -2 -- Vector<complex<float>> * Vector<complex<float>>\n"
      << "   -5 -- Vector<        float > * Vector<complex<float>>\n"
      << "\n"
      << "\ndouble-precision:\n"
      << "  (101-113)\n"
      << "  (131-132)\n"
      ;

  default:
    return 0;
  }
  return 1;
}
