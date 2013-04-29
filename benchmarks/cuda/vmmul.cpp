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
#include <vsip_csl/diagnostics.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::equal;

template <typename T,
	  typename Tag,
	  int      SD>
struct t_vmmul;

struct Impl_dev;		// Out-of-place vmmul, on-device 
                                //  (memory moves are not timed)


/***********************************************************************
  Impl_op: Out-of-place vmmul, on device
***********************************************************************/

template <typename T,
	  int      SD>
struct t_vmmul<T, Impl_dev, SD> : Benchmark_base
{
  typedef Map<Block_dist, Block_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;
  typedef Matrix<T, block_type>            view_type;

  typedef Dense<1, T, row1_type, Replicated_map<1> > replica_block_type;
  typedef Vector<T, replica_block_type>          replica_view_type;

  char const* what() { return "CUDA t_vmmul<T, Impl_dev, SD>"; }
  int ops(length_type rows, length_type cols)
    { return rows * cols * vsip::impl::Ops_info<T>::mul; }

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

    vsip_csl::profile::Timer t1;
    {    
      impl::cuda::dda::Data<replica_block_type, dda::in> dev_W(W.block());
      impl::cuda::dda::Data<block_type, dda::in> dev_A(A.block());
      impl::cuda::dda::Data<block_type, dda::out> dev_Z(Z.block());
      
      t1.start();
      if (SD == row)
      {
        for (index_type l=0; l<loop; ++l)
        {
          impl::cuda::vmmul_row(
            dev_W.ptr(),
            dev_A.ptr(),
            dev_Z.ptr(),
            rows,
            cols);
          cudaThreadSynchronize();
        }
      }
      else
      {
        for (index_type l=0; l<loop; ++l)
        {
          impl::cuda::vmmul_col(
            dev_W.ptr(),
            dev_A.ptr(),
            dev_Z.ptr(),
            rows,
            cols);
          cudaThreadSynchronize();
        }
      }
      t1.stop();
    }
    time = t1.delta();

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

    vsip_csl::assign_diagnostics(Z, vmmul<SD>(W, A));
  }
};



/***********************************************************************
  Fixed rows driver
***********************************************************************/

template <typename T, typename ImplTag, int SD>
struct t_vmmul_fix_rows : public t_vmmul<T, ImplTag, SD>
{
  typedef t_vmmul<T, ImplTag, SD> base_type;

  char const* what() { return SD == row ? "CUDA t_vmmul_fix_rows, by row" : 
                                          "CUDA t_vmmul_fix_rows, by col"; }
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

  char const* what() { return SD == row ? "CUDA t_vmmul_fix_cols, by row" : 
                                          "CUDA t_vmmul_fix_cols, by col"; }
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
test(Loop1P& loop, int what)
{
  length_type nr = atoi(loop.param_["rows"].c_str());
  length_type nc = atoi(loop.param_["cols"].c_str());

  switch (what)
  {
  case  1: loop(t_vmmul_fix_rows<complex<float>, Impl_dev,    row>(nr)); break;

  case 11: loop(t_vmmul_fix_cols<complex<float>, Impl_dev,    row>(nc)); break;

  case 21: loop(t_vmmul_fix_rows<complex<float>, Impl_dev,    col>(nr)); break;

  case 31: loop(t_vmmul_fix_cols<complex<float>, Impl_dev,    col>(nc)); break;

  case 0:
    std::cout
      << "CUDA vmmul -- vector-matrix multiply\n"
      << " Sweeping column size, vmmul<row>:\n"
      << "   -1 -- Out-of-place, complex\n"
      << " Sweeping row size, vmmul<row>:\n"
      << "  -11 -- Out-of-place, complex\n"
      << " Sweeping column size, vmmul<col>:\n"
      << "  -21 -- Out-of-place, complex\n"
      << " Sweeping row size, vmmul<col>:\n"
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

