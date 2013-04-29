/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for VSIPL matrix-matrix products.

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;


template <typename T> struct prod_traits;

template <>
struct prod_traits<float>
{
  typedef vsip_mview_f matrix_type;

  static matrix_type *create_matrix(length_type r, length_type c, 
    float const& value)
  {
    matrix_type *view = vsip_mcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
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

  static void prod(
    matrix_type const* a, vsip_mat_op a_op, 
    matrix_type const* b, vsip_mat_op b_op, 
    matrix_type* c)
  { 
    vsip_gemp_f(1, a, a_op, b, b_op, 0, c);
  }

  static bool valid(index_type i, matrix_type *output, float const &value)
  {
    vsip_scalar_f r = vsip_mget_f(output, 0, 0);
    return equal(r, value);
  }

};

template <>
struct prod_traits<std::complex<float> >
{
  typedef vsip_cmview_f matrix_type;

  static matrix_type *create_matrix(length_type r, length_type c, 
    std::complex<float> const& value)
  {
    matrix_type *view = vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
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

  static void prod(
    matrix_type const* a, vsip_mat_op a_op, 
    matrix_type const* b, vsip_mat_op b_op, 
    matrix_type* c)
  { 
    vsip_cgemp_f(vsip_cmplx_f(1, 0), a, a_op, b, b_op, vsip_cmplx_f(0, 0), c);
  }

  static bool valid(index_type i, matrix_type *output, std::complex<float> const &value)
  {
    vsip_cscalar_f c = vsip_cmget_f(output, 0, 0);
    return (equal(c.r, value.real()) && equal(c.i, value.imag()));
  }

};


// Matrix-matrix product benchmark class.

template <typename T>
struct t_prod1 : Benchmark_base
{
  typedef prod_traits<T> traits;

  char const* what() { return "t_prod1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    length_type N = M;
    length_type P = M;

    typename traits::matrix_type *A = traits::create_matrix(M, N, T(1));
    typename traits::matrix_type *B = traits::create_matrix(N, P, T(1));
    typename traits::matrix_type *Z = traits::create_matrix(M, P, T(1));

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::prod(A, VSIP_MAT_NTRANS, B, VSIP_MAT_NTRANS, Z);
    }
    t1.stop();
    
    time = t1.delta();
  }

  t_prod1() {}
};



// Matrix-matrix product (with hermetian) benchmark class.

template <typename T>
struct t_prodh1 : Benchmark_base
{
  typedef prod_traits<T> traits;

  char const* what() { return "t_prodh1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    length_type N = M;
    length_type P = M;

    typename traits::matrix_type *A = traits::create_matrix(M, N, T(1));
    typename traits::matrix_type *B = traits::create_matrix(N, P, T(1));
    typename traits::matrix_type *Z = traits::create_matrix(M, P, T(1));

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::prod(A, VSIP_MAT_NTRANS, B, VSIP_MAT_HERM, Z);
    }
    t1.stop();

    time = t1.delta();
  }

  t_prodh1() {}
};



// Matrix-matrix product (with tranpose) benchmark class.

template <typename T>
struct t_prodt1 : Benchmark_base
{
  typedef prod_traits<T> traits;

  char const* what() { return "t_prodt1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    length_type N = M;
    length_type P = M;

    typename traits::matrix_type *A = traits::create_matrix(M, N, T(1));
    typename traits::matrix_type *B = traits::create_matrix(N, P, T(1));
    typename traits::matrix_type *Z = traits::create_matrix(M, P, T(1));

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      traits::prod(A, VSIP_MAT_NTRANS, B, VSIP_MAT_TRANS, Z);
    }
    t1.stop();
    
    time = t1.delta();
  }

  t_prodt1() {}
};




void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.stop_  = 11;
}



int
test(Loop1P& loop, int what)
{
  using namespace vsip_csl::dispatcher;

  switch (what)
  {
  case  1: loop(t_prod1<float>()); break;
  case  2: loop(t_prod1<complex<float> >()); break;

  case  11: loop(t_prodt1<float>()); break;
  case  12: loop(t_prodt1<complex<float> >()); break;
  case  13: loop(t_prodh1<complex<float> >()); break;

  case    0:
    std::cout
      << "prod -- matrix-matrix product\n"
      << "    -1 -- default implementation, float\n"
      << "    -2 -- default implementation, complex<float>\n"
      << "   -11 -- default impl with transpose, float\n"
      << "   -12 -- default impl with transpose, complex<float>\n"
      << "   -13 -- default impl with hermetian, complex<float>\n"
      ;
  default: return 0;
  }
  return 1;
}
