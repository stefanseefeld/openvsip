/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/pwarp.cpp
    @author  Jules Bergmann
    @date    2007-11-19
    @brief   VSIPL++ Library: Perspective warp bridge with the CBE ALF.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define DEBUG 0

#if DEBUG
#  include <iostream>
#endif

#include <vsip/support.hpp>
#include <vsip/core/allocation.hpp>
#include <vsip/opt/cbe/pwarp_params.h>
#include <vsip/opt/cbe/ppu/pwarp.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/matrix.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cbe
{

// Transform coordinates (u, v) into (x, y) with projection matrix P.

template <typename T,
	  typename CoeffT,
	  typename Block1>
void
apply_proj(
  vsip::const_Matrix<CoeffT, Block1> P,
  T                                  u,
  T                                  v,
  T&                                 x,
  T&                                 y)
{
  T w =  u * P.get(2, 0) + v * P.get(2, 1) + P.get(2,2);
  x   = (u * P.get(0, 0) + v * P.get(0, 1) + P.get(0,2)) / w;
  y   = (u * P.get(1, 0) + v * P.get(1, 1) + P.get(1,2)) / w;
}



// Quantize value to smaller than or equal value that is multiple of quantum.

inline vsip::length_type
quantize_floor(vsip::length_type x, vsip::length_type quantum)
{
  // assert(quantum is power of 2);
  return x & ~(quantum-1);
}



// Quantize value to larger than or equal value that is multiple of quantum.

inline vsip::length_type
quantize_ceil(
  vsip::length_type x,
  vsip::length_type quantum,
  vsip::length_type max)
{
  // assert(quantum is power of 2);
  x = (x-1 % quantum == 0) ? x : (x & ~(quantum-1)) + quantum-1;
  if (x > max) x = max;
  return x;
}



length_type
pwarp_create_workblocks(
  Matrix<float> P,
  Pwarp_params& pwp,
  length_type   in_rows,
  length_type   in_cols,
  length_type   out_rows,
  length_type   out_cols,
  length_type   rows_per_spe,
  length_type   col_chunk_size,
  length_type   row_quantum,
  length_type   col_quantum,
  Task&         task,
  int           send_workblocks)
{
  typedef float CoeffT;

  using vsip::length_type;
  using vsip::index_type;
  using vsip::Domain;
  using std::min;
  using std::max;
  
  length_type max_size = 1;

  for (index_type r=0; r<out_rows; r += rows_per_spe)
  {
    length_type actual_rows = std::min(rows_per_spe, out_rows - r);

    for (index_type c=0; c<out_cols; c += col_chunk_size)
    {
      length_type actual_cols = std::min(col_chunk_size, out_cols-c);

      CoeffT u00, v00;
      CoeffT u01, v01;
      CoeffT u10, v10;
      CoeffT u11, v11;
      apply_proj<CoeffT>(P, c+0*actual_cols, r+0*actual_rows, u00, v00);
      apply_proj<CoeffT>(P, c+0*actual_cols, r+1*actual_rows, u01, v01);
      apply_proj<CoeffT>(P, c+1*actual_cols, r+0*actual_rows, u10, v10);
      apply_proj<CoeffT>(P, c+1*actual_cols, r+1*actual_rows, u11, v11);

      // Use ceilf(val-1) instead of floorf(val) (and floorf(val+1)
      // instead of ceilf(val)) to handle case when val is an integral
      // value.
      CoeffT min_u = min(min(u00-1, u01-1), min(u10-1, u11-1));
      CoeffT min_v = min(min(v00-1, v01-1), min(v10-1, v11-1));
      CoeffT max_u = max(max(u00, u01),max(u10, u11));
      CoeffT max_v = max(max(v00, v01),max(v10, v11));

      min_u = min(max(CoeffT(0), min_u), CoeffT(in_cols-1));
      max_u = min(max(CoeffT(0), max_u), CoeffT(in_cols-1));
      min_v = min(max(CoeffT(0), min_v), CoeffT(in_rows-1));
      max_v = min(max(CoeffT(0), max_v), CoeffT(in_rows-1));

      index_type in_r0, in_c0, in_r1, in_c1;
      in_r0 = quantize_floor((index_type)ceilf(min_v), row_quantum);
      in_c0 = quantize_floor((index_type)ceilf(min_u), col_quantum);
      in_r1 = quantize_ceil((index_type)floorf(max_v+1),row_quantum,in_rows-1);
      in_c1 = quantize_ceil((index_type)floorf(max_u+1),col_quantum,in_cols-1);

      pwp.in_row_0 = in_r0;
      pwp.in_col_0 = in_c0;
      pwp.in_rows  = in_r1 - in_r0 + 1;
      pwp.in_cols  = in_c1 - in_c0 + 1;

      pwp.out_row_0 = r;
      pwp.out_col_0 = c;
      pwp.out_rows  = actual_rows;
      pwp.out_cols  = actual_cols;

      if (pwp.in_rows * pwp.in_cols > max_size)
	max_size = pwp.in_rows * pwp.in_cols;

      if (send_workblocks)
      {
	assert(pwp.in_rows * pwp.in_cols <= VSIP_IMPL_CBE_PWARP_BUFFER_SIZE);

	Workblock block = task.create_workblock(actual_rows);
	block.set_parameters(pwp);
	block.enqueue();
      }

#if DEBUG >= 2
      std::cout << "CBE in 0: " << in_r0 << " " << in_c0 << "   "
		<< "1: " << in_r1 << " " << in_c1 << "   "
		<< "rows: " << (in_r1 - in_r0 + 1)
		<< std::endl;
      std::cout << "    u: " << min_u << " .. " << max_u << std::endl;
      std::cout << "    v: " << min_v << " .. " << max_v << std::endl;
/*
      std::cout << "    out 0: " << r << " " << c << "   "
		<< "rows: " << actual_rows << " " << actual_cols
		<< std::endl;*/
#endif
    }
  }

  return max_size;
}


// ALF bridge function for perspective warp.

template <typename T>
void
pwarp_block_impl(
  Matrix<float> P,
  T const*      p_in,
  stride_type   in_stride_0,
  T*            p_out,
  stride_type   out_stride_0,
  length_type   in_rows,
  length_type   in_cols,
  length_type   out_rows,
  length_type   out_cols)
{
  typedef float CoeffT;

  using vsip::length_type;
  using vsip::index_type;
  using vsip::Domain;
  using std::min;
  using std::max;

  length_type max_col_chunk_size = pwarp_block_max_col_size;

  length_type col_chunk_size = out_cols;
  length_type row_chunk_size = (128*128)/col_chunk_size;

  assert(col_chunk_size < max_col_chunk_size);

  length_type row_quantum = 1;
  length_type col_quantum = 128/sizeof(T);

  assert(is_dma_addr_ok(p_in));
  assert(is_dma_addr_ok(p_out));
  assert(is_dma_addr_ok(p_in  + in_stride_0));
  assert(is_dma_addr_ok(p_out + out_stride_0));

  Pwarp_params pwp;

  pwp.P[0] = P.get(0, 0);
  pwp.P[1] = P.get(0, 1);
  pwp.P[2] = P.get(0, 2);
  pwp.P[3] = P.get(1, 0);
  pwp.P[4] = P.get(1, 1);
  pwp.P[5] = P.get(1, 2);
  pwp.P[6] = P.get(2, 0);
  pwp.P[7] = P.get(2, 1);
  pwp.P[8] = P.get(2, 2);

  pwp.ea_in        = ea_from_ptr(p_in);
  pwp.ea_out       = ea_from_ptr(p_out);
  pwp.in_stride_0  = in_stride_0;
  pwp.out_stride_0 = out_stride_0;

  Task_manager *mgr = Task_manager::instance();
  Task task = mgr->reserve<Pwarp_tag, void(T,T)>
    (8*1024, // max stack size
     sizeof(Pwarp_params), 
     0,
     sizeof(T)*max_col_chunk_size,
     true);

  length_type spes         = mgr->num_spes();
  length_type rows_per_spe = min(out_rows / spes, row_chunk_size);
  // length_type n_wbs        = out_rows / rows_per_spe;

#if DEBUG
  std::cout << "CBE rows_per_spe: " << rows_per_spe << "\n"
	    << "  (max: " << row_chunk_size << ")"
	    << "\n";
// std::cout << "    ea_in: " << (unsigned long long)p_in 
//	    << "   " << pwp.ea_in << "\n";
#endif

  int is_ok = 0;

  // Plan 1. try to stream output image by row (keeping whole rows intact)
  while (is_ok == 0 && rows_per_spe >= 4)
  {
    length_type max_size = pwarp_create_workblocks(
      P,
      pwp,
      in_rows, in_cols,
      out_rows, out_cols,
      rows_per_spe, col_chunk_size,
      row_quantum, col_quantum,
      task, 0);
    if (max_size <= VSIP_IMPL_CBE_PWARP_BUFFER_SIZE)
      is_ok = 1;
    else 
      rows_per_spe /= 2;
  }

  // Plan B. stream output image by row/col.
  if (is_ok == 0)
  {
    rows_per_spe = 64;
    col_chunk_size = 64;
    while (is_ok == 0 && rows_per_spe >= 4)
    {
      length_type max_size = pwarp_create_workblocks(
	P,
	pwp,
	in_rows, in_cols,
	out_rows, out_cols,
	rows_per_spe, col_chunk_size,
	row_quantum, col_quantum,
	task, 0);

      if (max_size <= VSIP_IMPL_CBE_PWARP_BUFFER_SIZE)
	is_ok = 1;
      else 
	rows_per_spe /= 2;
    }
  }

  if (is_ok)
    (void)pwarp_create_workblocks(
      P,
      pwp,
      in_rows, in_cols,
      out_rows, out_cols,
      rows_per_spe, col_chunk_size,
      row_quantum, col_quantum,
      task, 1);
  else
    assert(0);

  task.sync();
}



template
void
pwarp_block_impl(
  Matrix<float> P,
  unsigned char const* in,
  stride_type          in_stride_0,
  unsigned char*       out,
  stride_type          out_stride_0,
  length_type          in_rows,
  length_type          in_cols,
  length_type          out_rows,
  length_type          out_cols);

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
