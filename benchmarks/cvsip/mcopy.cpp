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
using vsip_csl::equal;


template <typename T> struct mcopy_traits;

template <>
struct mcopy_traits<float>
{
  typedef vsip_mview_f matrix_type;

  static void initialize()
  { vsip_init((void *)0); }

  static matrix_type *create_matrix(length_type r, length_type c, 
    float const& value, vsip_major major = VSIP_ROW)
  {
    matrix_type *view = vsip_mcreate_f(r, c, major, VSIP_MEM_NONE);
    vsip_mfill_f(value, view);
    return view;
  }
  static void delete_matrix(matrix_type *m) { return vsip_malldestroy_f(m); }

  static void get(index_type i, index_type j, matrix_type *output, float& value)
  {
    value = vsip_mget_f(output, i, j);
  }
  static void put(index_type i, index_type j, matrix_type *input, float value)
  {
    vsip_mput_f(input, i, j, value);
  }

  static void mcopy(matrix_type const* a, matrix_type* b)
  { 
    vsip_mcopy_f_f(a, b);
  }

  static void mtrans(matrix_type const* a, matrix_type* b)
  { 
    vsip_mtrans_f(a, b);
  }

  static bool valid(index_type i, index_type j, matrix_type *output, float const &value)
  {
    vsip_scalar_f r = vsip_mget_f(output, i, j);
    return equal(r, value);
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};

template <>
struct mcopy_traits<std::complex<float> >
{
  typedef vsip_cmview_f matrix_type;

  static void initialize()
  { vsip_init((void *)0); }

  static matrix_type *create_matrix(length_type r, length_type c, 
    std::complex<float> const& value, vsip_major major = VSIP_ROW)
  {
    matrix_type *view = vsip_cmcreate_f(r, c, major, VSIP_MEM_NONE);
    vsip_cmfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static void delete_matrix(matrix_type *m) { return vsip_cmalldestroy_f(m); }

  static void get(index_type i, index_type j, matrix_type *output, std::complex<float>& value)
  {
    vsip_cscalar_f cval;
    cval = vsip_cmget_f(output, i, j);
    value.real() = cval.r;
    value.imag() = cval.i;
  }
  static void put(index_type i, index_type j, matrix_type *input, std::complex<float> value)
  {
    vsip_cscalar_f cval;
    cval.r = value.real();
    cval.i = value.imag();
    vsip_cmput_f(input, i, j, cval);
  }

  static void mcopy(matrix_type const* a, matrix_type* b)
  { 
    vsip_cmcopy_f_f(a, b);
  }

  static void mtrans(matrix_type const* a, matrix_type* b)
  { 
    vsip_cmtrans_f(a, b);
  }

  static bool valid(index_type i, index_type j, matrix_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cmget_f(output, i, j);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};


/***********************************************************************
  Matrix copy - normal assignment
***********************************************************************/

template <typename T,
	  typename SrcOrder,
	  typename DstOrder>
struct t_mcopy : Benchmark_base
{
  typedef mcopy_traits<T> traits;

  char const* what() { return "t_mcopy<T, SrcOrder, DstOrder>"; }
  int ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size) { return size*sizeof(T); }
  int wiob_per_point(length_type size) { return size*sizeof(T); }
  int mem_per_point(length_type size)  { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    length_type const M = size;
    length_type const N = size;

    typename traits::matrix_type *A = traits::create_matrix(M, N, T(M_PI));
    typename traits::matrix_type *B = traits::create_matrix(M, N, T());
    traits::put(size/2,      0, A, M_PI / 2);
    traits::put(     0, size/2, A, M_PI / 4);

    vsip_csl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      if (vsip_csl::is_same<SrcOrder, DstOrder>::value)
        traits::mcopy(A, B);
      else
        traits::mtrans(A, B);
    }
    t1.stop();


    // If transposed, the expected values are swapped.
    T check_1 = vsip_csl::is_same<SrcOrder, DstOrder>::value ? M_PI/2 : M_PI/4;
    T check_2 = vsip_csl::is_same<SrcOrder, DstOrder>::value ? M_PI/4 : M_PI/2;

    if (!traits::valid(0,      0,      B, M_PI) ||
        !traits::valid(size/2, 0,      B, check_1) ||
        !traits::valid(0,      size/2, B, check_2) ||
        !traits::valid(size-1, size-1, B, M_PI))
      std::cout << "t_mcopy: ERROR" << std::endl;

    time = t1.delta();
  }

  t_mcopy()
  {
    traits::initialize();
  }

  ~t_mcopy()
  {
    traits::finalize();
  }
};



/***********************************************************************
  Main functions
***********************************************************************/

void
defaults(Loop1P& loop)
{
  loop.stop_ = 12;
}



int
test(Loop1P& loop, int what)
{
  processor_type np = num_processors();

  typedef row2_type rt;
  typedef col2_type ct;

  switch (what)
  {
  case  1: loop(t_mcopy<float, rt, rt>()); break;
  case  2: loop(t_mcopy<float, rt, ct>()); break;
  case  3: loop(t_mcopy<float, ct, rt>()); break;
  case  4: loop(t_mcopy<float, ct, ct>()); break;

  case  5: loop(t_mcopy<complex<float>, rt, rt>()); break;
  case  6: loop(t_mcopy<complex<float>, rt, ct>()); break;
  case  7: loop(t_mcopy<complex<float>, ct, rt>()); break;
  case  8: loop(t_mcopy<complex<float>, ct, ct>()); break;

  case   0:
    std::cout
      << "mcopy -- matrix copy with and without transpose\n"
      << "    -1:         float,  rows <- rows, assignment\n"
      << "    -2:         float,  rows <- cols, assignment\n"
      << "    -3:         float,  cols <- rows, assignment\n"
      << "    -4:         float,  cols <- cols, assignment\n"
      << "    -5: complex<float>, rows <- rows, assignment\n"
      << "    -6: complex<float>, rows <- cols, assignment\n"
      << "    -7: complex<float>, cols <- rows, assignment\n"
      << "    -8: complex<float>, cols <- cols, assignment\n"
      << std::endl;

  default:
    return 0;
  }
  return 1;
}
