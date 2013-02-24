/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL benchmark for Vector-Matrix Multiply

// Originally created 2010-06-24 by Don McCoy

#include "../benchmarks.hpp"
#include <vsip.h>
#include <iostream>

using namespace vsip;



// This traits class allows the benchmark code to be written generically
// (once, that is) because it abstracts all the vsipl calls into these
// type-dependent wrappers.

template <typename T1> struct vmmul_traits;

template <>
struct vmmul_traits<float>
{
  typedef vsip_vview_f vector_type;
  typedef vsip_mview_f matrix_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, float const& value)
  {
    vector_type *view = vsip_vcreate_f(l, VSIP_MEM_NONE);
    vsip_vfill_f(value, view);
    return view;
  }
  static matrix_type *create_matrix(length_type r, length_type c,
    float const& value)
  {
    matrix_type *view = vsip_mcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_mfill_f(value, view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_valldestroy_f(v); }
  static void delete_matrix(matrix_type *m) { vsip_malldestroy_f(m); }

  static void vmmul(vector_type const *v, matrix_type const *m, vsip_major sd, 
    matrix_type *r)
  { vsip_vmmul_f(v, m, sd, r); }

  static bool equal(vector_type const *v, matrix_type const *m, vsip_major sd,
    matrix_type const *r, index_type i, index_type j)
  {
    index_type idx = (sd == VSIP_ROW ? j : i);
    return ::equal(vsip_vget_f(v, idx) * vsip_mget_f(m, i, j), vsip_mget_f(r, i, j));
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};


template <>
struct vmmul_traits<std::complex<float> >
{
  typedef vsip_cvview_f vector_type;
  typedef vsip_cmview_f matrix_type;

  static void initialize()
  { vsip_init((void *)0); }

  static vector_type *create_vector(length_type l, 
    std::complex<float> const& value)
  {
    vector_type *view = vsip_cvcreate_f(l, VSIP_MEM_NONE);
    vsip_cvfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static matrix_type *create_matrix(length_type r, length_type c, 
    std::complex<float> const& value)
  {
    matrix_type *view = vsip_cmcreate_f(r, c, VSIP_ROW, VSIP_MEM_NONE);
    vsip_cmfill_f(vsip_cmplx_f(value.real(), value.imag()), view);
    return view;
  }
  static void delete_vector(vector_type *v) { vsip_cvalldestroy_f(v); }
  static void delete_matrix(matrix_type *m) { vsip_cmalldestroy_f(m); }

  static void vmmul(vector_type const *v, matrix_type const *m, vsip_major sd, 
    matrix_type *r)
  { vsip_cvmmul_f(v, m, sd, r); }

  static bool equal(vector_type const *v, matrix_type const *m, vsip_major sd, 
    matrix_type const *r,
    index_type i, index_type j)
  {
    index_type idx = (sd == VSIP_ROW ? j : i);
    vsip_cscalar_f lhs = vsip_cmul_f(vsip_cvget_f(v, idx), vsip_cmget_f(m, i, j));
    vsip_cscalar_f rhs = vsip_cmget_f(r, i, j);
    return ::equal(lhs.r, rhs.r) && ::equal(lhs.i, rhs.i);
  }

  static void finalize()
  { vsip_finalize((void *)0); }
};




template <typename T,
	  int      SD>
struct t_vmmul : Benchmark_base
{
  typedef vmmul_traits<T> traits;

  int ops(length_type rows, length_type cols)
    { return rows * cols * vsip::impl::Ops_info<T>::mul; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typename traits::vector_type *A = traits::create_vector(SD == row ? cols : rows, T(2));
    typename traits::matrix_type *B = traits::create_matrix(rows, cols, T(3));
    typename traits::matrix_type *C = traits::create_matrix(rows, cols, T());


    vsip::impl::profile::Timer t1;
    vsip_major sd = (SD == vsip::row ? VSIP_ROW : VSIP_COL);
    
    t1.start();
    for (index_type l = 0; l < loop; ++l)
    {
      traits::vmmul(A, B, sd, C);
    }
    t1.stop();
    
    for (index_type i = 0; i < rows; ++i)
      for (index_type j = 0; j < cols; ++j)
        test_assert(traits::equal(A, B, sd, C, i, j));
    
    time = t1.delta();
    
    traits::delete_matrix(C);
    traits::delete_matrix(B);
    traits::delete_vector(A);
  }

  t_vmmul() { traits::initialize(); }
  ~t_vmmul() { traits::finalize(); }
};



//  Fixed rows driver

template <typename T,
          int      SD>
struct t_vmmul_fix_rows : public t_vmmul<T, SD>
{
  typedef t_vmmul<T, SD> base_type;

  char const* what() { return "t_vmmul_fix_rows"; }
  float ops_per_point(length_type cols)
    { return this->ops(rows_, cols) / cols; }

  int riob_per_point(length_type cols)
    { return SD == row ? (rows_+1         )*sizeof(T)
                       : (rows_+rows_/cols)*sizeof(T); }

  int wiob_per_point(length_type cols)
    { return SD == row ? (rows_+1         )*sizeof(T)
                       : (rows_+rows_/cols)*sizeof(T); }

  int mem_per_point(length_type cols)
  { return SD == row ? (2*rows_+1)*sizeof(T)
                     : (2*rows_+rows_/cols)*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    this->exec(rows_, cols, loop, time);
  }

  t_vmmul_fix_rows(length_type rows) : rows_(rows) {}

// Member data
  length_type rows_;
};




//  Fixed cols driver

template <typename T,
          int      SD>
struct t_vmmul_fix_cols : public t_vmmul<T, SD>
{
  typedef t_vmmul<T, SD> base_type;

  char const* what() { return "t_vmmul_fix_cols"; }
  float ops_per_point(length_type rows)
    { return this->ops(rows, cols_) / rows; }

  int riob_per_point(length_type rows)
    { return SD == row ? (cols_+cols_/rows)*sizeof(T)
                       : (cols_+1         )*sizeof(T); }

  int wiob_per_point(length_type rows)
    { return SD == row ? (cols_+cols_/rows)*sizeof(T)
                       : (cols_+1         )*sizeof(T); }

  int mem_per_point(length_type rows)
  { return SD == row ? (2*cols_+cols_/rows)*sizeof(T)
                     : (2*cols_+1)*sizeof(T); }

  void operator()(length_type rows, length_type loop, float& time)
  {
    this->exec(rows, cols_, loop, time);
  }

  t_vmmul_fix_cols(length_type cols) : cols_(cols) {}

// Member data
  length_type cols_;
};


void
defaults(Loop1P& loop)
{
  loop.start_      = 4;
  loop.stop_       = 16;
  loop.loop_start_ = 10;
  loop.user_param_ = 256;

  loop.param_["rows"] = "64";
  loop.param_["cols"] = "2048";
}


int
test(Loop1P& loop, int what)
{
  length_type nr = atoi(loop.param_["rows"].c_str());
  length_type nc = atoi(loop.param_["cols"].c_str());

  switch (what)
  {
  case  1: loop(t_vmmul_fix_rows<complex<float>, row>(nr)); break;
  case 11: loop(t_vmmul_fix_cols<complex<float>, row>(nc)); break;
  case 21: loop(t_vmmul_fix_rows<complex<float>, col>(nr)); break;
  case 31: loop(t_vmmul_fix_cols<complex<float>, col>(nc)); break;

  case 0:
    std::cout
      << "vmmul -- vector-matrix multiply\n"
      << " Sweeping number of columns, row-wise orientation:\n"
      << "   -1 -- Out-of-place, complex\n"
      << " Sweeping number of rows, row-wise orientation:\n"
      << "  -11 -- Out-of-place, complex\n"
      << " Sweeping number of columns, column-wise orientation:\n"
      << "  -21 -- Out-of-place, complex\n"
      << " Sweeping number of rows, column-wise orientation:\n"
      << "  -31 -- Out-of-place, complex\n"
      << "\n"
      << " Parameters (for sweeping number of columns, cases 1, 21)\n"
      << "  -p:rows ROWS -- set number of rows (default 64)\n"
      << "\n"
      << " Parameters (for sweeping number of columns, cases 11, 31)\n"
      << "  -p:cols COLS -- set number of columns (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}
