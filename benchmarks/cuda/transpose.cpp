/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for vector-matrix multiply using CUDA.

#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <vsip/opt/cuda/kernels.hpp>

#include <vsip_csl/output.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::equal;

template <typename T,
	  typename Tag>
struct t_transpose;

struct Impl_dev;		// Out-of-place transpose, on-device 
                                //  (memory moves are not timed)

struct Impl_view;		// Out-of-place transpose, normal
                                //  (memory moves are timed)

/***********************************************************************
  Impl_dev: Out-of-place transpose, on device
***********************************************************************/

template <typename T>
struct t_transpose<T, Impl_dev> : Benchmark_base
{
  typedef Dense<2, T, row2_type> block_type;
  typedef Matrix<T, block_type>  view_type;

  char const* what() { return "CUDA transpose (on device)"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    view_type         A(cols, rows);
    view_type         Z(rows, cols, T());

    // Set up a ramp pattern so that each location has a unique value
    for (index_type i = 0; i < cols; ++i)
      A.row(i) = ramp(T(i*rows+1), T(1), A.size(1));

    vsip_csl::profile::Timer t1;
    {    
      impl::cuda::dda::Data<block_type, dda::in> dev_A(A.block());
      impl::cuda::dda::Data<block_type, dda::out> dev_Z(Z.block());
      
      t1.start();
      for (index_type l=0; l<loop; ++l)
      {
        impl::cuda::transpose(
          dev_A.ptr(),
          dev_Z.ptr(),
          rows,
          cols);
      }
      t1.stop();
    }
    time = t1.delta();

    for (index_type r = 0; r < rows; ++r)
      for (index_type c = 0; c < cols; ++c)
        test_assert(equal(Z.get(r, c), A.get(c, r)));
  }
};



/***********************************************************************
  Impl_view: Out-of-place transpose, using view operators
***********************************************************************/

template <typename T>
struct t_transpose<T, Impl_view> : Benchmark_base
{
  typedef Dense<2, T, row2_type> block_type;
  typedef Matrix<T, block_type>  view_type;

  char const* what() { return "CUDA transpose (Dense)"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols; }

  void exec(length_type rows, length_type cols, length_type loop, float& time)
  {
    view_type         A(cols, rows);
    view_type         Z(rows, cols, T());

    // Set up a ramp pattern so that each location has a unique value
    for (index_type i = 0; i < cols; ++i)
      A.row(i) = ramp(T(i*rows+1), T(1), A.size(1));

    vsip_csl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      Z = A.transpose();
    }
    t1.stop();
    time = t1.delta();

    for (index_type r = 0; r < rows; ++r)
      for (index_type c = 0; c < cols; ++c)
        test_assert(equal(Z.get(r, c), A.get(c, r)));
  }
};


/***********************************************************************
  Fixed rows driver
***********************************************************************/

template <typename T, typename ImplTag>
struct t_transpose_fix_rows : public t_transpose<T, ImplTag>
{
  typedef t_transpose<T, ImplTag> base_type;

  float ops_per_point(length_type /*cols*/)  { return rows_; }
  int riob_per_point(length_type /*cols*/)   { return rows_ * sizeof(T); }
  int wiob_per_point(length_type /*cols*/)   { return rows_ * sizeof(T); }
  int mem_per_point(length_type /*cols*/)    { return rows_ * sizeof(T) * 2; }

  void operator()(length_type cols, length_type loop, float& time)
  {
    this->exec(rows_, cols, loop, time);
  }

  t_transpose_fix_rows(length_type rows) : rows_(rows) {}

// Member data
  length_type rows_;
};



/***********************************************************************
  Fixed cols driver
***********************************************************************/

template <typename T, typename ImplTag>
struct t_transpose_fix_cols : public t_transpose<T, ImplTag>
{
  typedef t_transpose<T, ImplTag> base_type;

  float ops_per_point(length_type /*rows*/)  { return cols_; }
  int riob_per_point(length_type /*rows*/)   { return cols_ * sizeof(T); }
  int wiob_per_point(length_type /*rows*/)   { return cols_ * sizeof(T); }
  int mem_per_point(length_type /*rows*/)    { return cols_ * sizeof(T) * 2; }

  void operator()(length_type rows, length_type loop, float& time)
  {
    this->exec(rows, cols_, loop, time);
  }

  t_transpose_fix_cols(length_type cols) : cols_(cols) {}

// Member data
  length_type cols_;
};



/***********************************************************************
  Sweep rows and columns together (square matrices)
***********************************************************************/

template <typename T, typename ImplTag>
struct t_transpose_square : public t_transpose<T, ImplTag>
{
  typedef t_transpose<T, ImplTag> base_type;

  float ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size)   { return size * sizeof(T); }
  int wiob_per_point(length_type size)   { return size * sizeof(T); }
  int mem_per_point(length_type size)    { return size * sizeof(T) * 2; }

  void operator()(length_type size, length_type loop, float& time)
  {
    this->exec(size, size, loop, time);
  }

  t_transpose_square() {}
};


void
defaults(Loop1P& loop)
{
  loop.start_      = 4;
  loop.stop_       = 15;
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
  case 1: loop(t_transpose_fix_rows<complex<float>, Impl_dev>(nr)); break;
  case 2: loop(t_transpose_fix_rows<complex<float>, Impl_view>(nr)); break;

  case 11: loop(t_transpose_fix_cols<complex<float>, Impl_dev>(nc)); break;
  case 12: loop(t_transpose_fix_cols<complex<float>, Impl_view>(nc)); break;

  case 21: loop(t_transpose_square<complex<float>, Impl_dev>()); break;
  case 22: loop(t_transpose_square<complex<float>, Impl_view>()); break;


  case 0:
    std::cout
      << "CUDA transpose sweeping column size:\n"
      << "    -1 -- Out-of-place, complex (direct - memory moves not timed)\n"
      << "    -2 -- Out-of-place, complex (normal - using Dense blocks)\n"
      << "\n"
      << "CUDA transpose sweeping row size:\n"
      << "   -11 -- Out-of-place, complex (direct - memory moves not timed)\n"
      << "   -12 -- Out-of-place, complex (normal - using Dense blocks)\n"
      << "\n"
      << "CUDA transpose using square matrices:\n"
      << "   -21 -- Out-of-place, complex (direct - memory moves not timed)\n"
      << "   -22 -- Out-of-place, complex (normal - using Dense blocks)\n"
      << "\n"
      << " Parameters (for sweeping number of rows / column size, cases 1 & 2)\n"
      << "  -p:rows ROWS -- set number of rows (default 64)\n"
      << "\n"
      << " Parameters (for sweeping number of columns / row size, cases 11 & 12)\n"
      << "  -p:cols COLS -- set number of columns (default 2048)\n"
      ;

  default: return 0;
  }
  return 1;
}

