//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for vector-matrix multiply.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include "benchmark.hpp"

using namespace ovxx;

template <typename T,
	  typename Tag,
	  int      SD>
struct t_vmmul;

struct Impl_op;		// Out-of-place vmmul
struct Impl_pop;	// Psuedo out-of-place vmmul, using vector-multiply
struct Impl_s_op;	// Scaled, Out-of-place vmmul
struct Impl_s_pop;	// Scaled, psuedo out-of-place vmmul, using vmul



/***********************************************************************
  Impl_op: Out-of-place vmmul
***********************************************************************/
#if OVXX_PARALLEL_API == 1
template <typename T,
	  int      SD>
struct t_vmmul<T, Impl_op, SD> : Benchmark_base
{
  typedef Map<Block_dist, Block_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  typedef Dense<1, T, row1_type, Replicated_map<1> > replica_block_type;
  typedef Vector<T, replica_block_type>          replica_view_type;

  char const* what() { return "t_vmmul<T, Impl_op, SD>"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols * ovxx::ops_count::traits<T>::mul; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Block_dist(1));

    replica_view_type W(SD == row ? cols : rows);
    view_type         A(rows, cols, T(), map);
    view_type         Z(rows, cols, map);

    Rand<T> rand(0);

    W = ramp(T(1), T(1), W.size());
    A = rand.randu(rows, cols);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = vmmul<SD>(W, A);
    time = t1.elapsed();
    
    if (SD == row)
    {
      length_type l_rows  = Z.local().size(0);

      for (index_type r=0; r<l_rows; ++r)
	for (index_type c=0; c<cols; ++c)
	  test_assert(equal(Z.local().get(r, c),
			    W.get(c) * A.local().get(r, c)));
    }
    else
    {
      length_type l_cols  = Z.local().size(1);
      for (index_type c=0; c<l_cols; ++c)
	for (index_type r=0; r<rows; ++r)
	  test_assert(equal(Z.local().get(r, c),
			    W.get(r) * A.local().get(r, c)));
    }
  }

  void diag()
  {
    length_type const rows = 32;
    length_type const cols = 256;

    Vector<T>   W(SD == row ? cols : rows);
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);
    // TBD
  }
};
#endif


/***********************************************************************
  Impl_pop: Psuedo out-of-place vmmul, using vector-multiply
***********************************************************************/

template <typename T,
	  int      SD>
struct t_vmmul<T, Impl_pop, SD> : Benchmark_base
{
  char const* what() { return "t_vmmul<T, Impl_op, SD>"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols * ovxx::ops_count::traits<T>::mul; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    Vector<T>   W(SD == row ? cols : rows);
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);

    W = ramp(T(1), T(1), W.size());
    A = T(1);
    
    if (SD == row)
    {
      timer t1;
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<rows; ++i)
	  Z.row(i) = W * A.row(i);
      time = t1.elapsed();
    }
    else
    {
      timer t1;
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<cols; ++i)
	  Z.col(i) = W * A.col(i);
      time = t1.elapsed();
    }
    
    test_assert(equal(Z(0, 0), T(1)));
  }
};



/***********************************************************************
  Impl_s_op: Scaled out-of-place vmmul
***********************************************************************/

template <typename T,
	  int      SD>
struct t_vmmul<T, Impl_s_op, SD> : Benchmark_base
{
  char const* what() { return "t_vmmul<T, Impl_s_op, SD>"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols * ovxx::ops_count::traits<T>::mul; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    Vector<T>   W(SD == row ? cols : rows);
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);

    W = ramp(T(1), T(1), W.size());
    A = T(1);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = T(2) * vmmul<SD>(W, A);
    time = t1.elapsed();
    
    test_assert(equal(Z(0, 0), T(2)));
  }
};



/***********************************************************************
  Impl_s_pop: Scaled, psuedo out-of-place vmmul, using vector-multiply
***********************************************************************/

template <typename T,
	  int      SD>
struct t_vmmul<T, Impl_s_pop, SD> : Benchmark_base
{
  char const* what() { return "t_vmmul<T, Impl_s_pop, SD>"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols * ovxx::ops_count::traits<T>::mul; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    Vector<T>   W(SD == row ? cols : rows);
    Matrix<T>   A(rows, cols, T());
    Matrix<T>   Z(rows, cols);

    W = ramp(T(1), T(1), W.size());
    A = T(1);
    
    if (SD == row)
    {
      timer t1;
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<rows; ++i)
	  Z.row(i) = T(2) * W * A.row(i);
      time = t1.elapsed();
    }
    else
    {
      timer t1;
      for (index_type l=0; l<loop; ++l)
	for (index_type i=0; i<cols; ++i)
	  Z.col(i) = T(2) * W * A.col(i);
      time = t1.elapsed();
    }
    
    test_assert(equal(Z(0, 0), T(2)));
  }
};



/***********************************************************************
  Fixed rows driver
***********************************************************************/

template <typename T, typename ImplTag, int SD>
struct t_vmmul_fix_rows : public t_vmmul<T, ImplTag, SD>
{
  typedef t_vmmul<T, ImplTag, SD> base_type;

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



/***********************************************************************
  Fixed cols driver
***********************************************************************/

template <typename T, typename ImplTag, int SD>
struct t_vmmul_fix_cols : public t_vmmul<T, ImplTag, SD>
{
  typedef t_vmmul<T, ImplTag, SD> base_type;

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



/***********************************************************************
  Fixed cols driver
***********************************************************************/


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
benchmark(Loop1P& loop, int what)
{
  length_type nr = atoi(loop.param_["rows"].c_str());
  length_type nc = atoi(loop.param_["cols"].c_str());

  switch (what)
  {
#if OVXX_PARALLEL_API == 1
  case  1: loop(t_vmmul_fix_rows<complex<float>, Impl_op,    row>(nr)); break;
#endif
  case  2: loop(t_vmmul_fix_rows<complex<float>, Impl_pop,   row>(nr)); break;
  case  3: loop(t_vmmul_fix_rows<complex<float>, Impl_s_op,  row>(nr)); break;
  case  4: loop(t_vmmul_fix_rows<complex<float>, Impl_s_pop, row>(nr)); break;

#if OVXX_PARALLEL_API == 1
  case 11: loop(t_vmmul_fix_cols<complex<float>, Impl_op,    row>(nc)); break;
#endif
  case 12: loop(t_vmmul_fix_cols<complex<float>, Impl_pop,   row>(nc)); break;
  case 13: loop(t_vmmul_fix_cols<complex<float>, Impl_s_op,  row>(nc)); break;
  case 14: loop(t_vmmul_fix_cols<complex<float>, Impl_s_pop, row>(nc)); break;

#if OVXX_PARALLEL_API == 1
  case 21: loop(t_vmmul_fix_rows<complex<float>, Impl_op,    col>(nr)); break;
#endif
  case 22: loop(t_vmmul_fix_rows<complex<float>, Impl_pop,   col>(nr)); break;
#if OVXX_PARALLEL_API == 1
  case 23: loop(t_vmmul_fix_rows<complex<float>, Impl_s_op,  col>(nr)); break;
#endif
  case 24: loop(t_vmmul_fix_rows<complex<float>, Impl_s_pop, col>(nr)); break;

#if OVXX_PARALLEL_API == 1
  case 31: loop(t_vmmul_fix_cols<complex<float>, Impl_op,    col>(nc)); break;
#endif
  case 32: loop(t_vmmul_fix_cols<complex<float>, Impl_pop,   col>(nc)); break;
  case 33: loop(t_vmmul_fix_cols<complex<float>, Impl_s_op,  col>(nc)); break;
  case 34: loop(t_vmmul_fix_cols<complex<float>, Impl_s_pop, col>(nc)); break;

  case 0:
    std::cout
      << "vmmul -- vector-matrix multiply\n"
      << " Sweeping number of columns, row major:\n"
      << "   -1 -- Out-of-place, complex\n"
      << "   -2 -- Out-of-place, complex, using vmul\n"
      << "   -3 -- Out-of-place, complex, scaled\n"
      << "   -4 -- Out-of-place, complex, scaled, using vmul\n"
      << " Sweeping number of rows, row major:\n"
      << "  -11 -- Out-of-place, complex\n"
      << "  -12 -- Out-of-place, complex, using vmul\n"
      << "  -13 -- Out-of-place, complex, scaled\n"
      << "  -14 -- Out-of-place, complex, scaled, using vmul\n"
      << " Sweeping number of columns, column major:\n"
      << "  -21 -- Out-of-place, complex\n"
      << "  -22 -- Out-of-place, complex, using vmul\n"
      << "  -23 -- Out-of-place, complex, scaled\n"
      << "  -24 -- Out-of-place, complex, scaled, using vmul\n"
      << " Sweeping number of rows, column major:\n"
      << "  -31 -- Out-of-place, complex\n"
      << "  -32 -- Out-of-place, complex, using vmul\n"
      << "  -33 -- Out-of-place, complex, scaled\n"
      << "  -34 -- Out-of-place, complex, scaled, using vmul\n"
      << "\n"
      << " Parameters (for sweeping number of columns, cases 1-4, 21-24)\n"
      << "  -p:rows ROWS -- set number of rows (default 64)\n"
      << "\n"
      << " Parameters (for sweeping number of columns, cases 11-14, 31-34)\n"
      << "  -p:cols COLS -- set number of columns (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}
