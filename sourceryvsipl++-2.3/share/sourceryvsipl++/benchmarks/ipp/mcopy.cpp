/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/mcopy_ipp.cpp
    @author  Jules Bergmann
    @date    2005-02-06
    @brief   VSIPL++ Library: Benchmark for matrix copy using IPP

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <ipp.h>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/core/parallel/assign_chain.hpp>
#include <vsip/map.hpp>
#include <vsip/opt/profile.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/ops_info.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/plainblock.hpp>
#include "loop.hpp"


using namespace vsip;
using vsip_csl::equal;
using vsip_csl::operator<<;
using vsip::impl::ITE_Type;
using vsip::impl::As_type;


void
ipp_copy(
  float*      src,
  stride_type src_row_stride,
  stride_type src_col_stride,
  float*      dst,
  stride_type dst_row_stride,
  stride_type dst_col_stride,
  length_type rows,
  length_type cols)
{
  ippmCopy_ma_32f_SS(
		src, rows*cols,
		src_row_stride*sizeof(float), src_col_stride*sizeof(float),
		dst, rows*cols,
		dst_row_stride*sizeof(float), dst_col_stride*sizeof(float),
		cols, rows,  1);
}



void
ipp_transpose(
  float*      src,
  stride_type src_row_stride,
  stride_type src_col_stride,
  float*      dst,
  stride_type dst_row_stride,
  stride_type dst_col_stride,
  length_type rows,
  length_type cols)
{
  assert(src_col_stride == 1);
  assert(dst_col_stride == 1);
#if IPP_VERSION_MAJOR >= 5
  IppStatus status = ippmTranspose_m_32f(
		src, src_row_stride*sizeof(float), sizeof(float),
		cols, rows,
		dst, dst_row_stride*sizeof(float), sizeof(float)
		);
#else
  // IPP 4.1
  IppStatus status = ippmTranspose_m_32f(src, src_row_stride*sizeof(float),
					 cols, rows,
					 dst, dst_row_stride*sizeof(float));
#endif
  test_assert(status == ippStsNoErr);
}



/***********************************************************************
  Matrix copy - normal assignment
***********************************************************************/

template <typename T,
	  typename DstBlock,
	  typename SrcBlock>
void
matrix_copy(
  Matrix<T, DstBlock> dst,
  Matrix<T, SrcBlock> src)
{
  dst = src;
}

struct Impl_copy;
struct Impl_transpose;
struct Impl_select;

template <typename T,
	  typename SrcBlock,
	  typename DstBlock,
	  typename ImplTag>
struct t_mcopy;



/***********************************************************************
  Matrix copy - using Copy 
***********************************************************************/

template <typename T,
	  typename SrcBlock,
	  typename DstBlock>
struct t_mcopy<T, SrcBlock, DstBlock, Impl_copy> : Benchmark_base
{
  typedef typename SrcBlock::map_type src_map_type;
  typedef typename DstBlock::map_type dst_map_type;

  char const* what() { return "t_mcopy<T, SrcBlock, DstBlock, Impl_copy>"; }
  int ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size) { return size*sizeof(T); }
  int wiob_per_point(length_type size) { return size*sizeof(T); }
  int mem_per_point(length_type size)  { return 2*size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    length_type const M = size;
    length_type const N = size;

    Matrix<T, SrcBlock>   A(M, N, T(), src_map_);
    Matrix<T, DstBlock>   Z(M, N,      dst_map_);

    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	A.put(m, n, T(m*N + n));
      }
    
    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<SrcBlock> src_ext(A.block());
      impl::Ext_data<DstBlock> dst_ext(Z.block());
    
      t1.start();
      for (index_type l=0; l<loop; ++l)
	ipp_copy(src_ext.data(), src_ext.stride(0), src_ext.stride(1),
		 dst_ext.data(), dst_ext.stride(0), dst_ext.stride(1),
		 M, N);
      t1.stop();
    }
    
    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	if (!equal(Z.get(m, n), T(m*N+n)))
	{
	  std::cout << "t_mcopy: ERROR" << std::endl;
	  std::cout << "A = " << A << std::endl;
	  std::cout << "Z = " << Z << std::endl;
	  abort();
	}
      }
    
    time = t1.delta();
  }

  t_mcopy(src_map_type src_map, dst_map_type dst_map)
    : src_map_(src_map),
      dst_map_(dst_map)
    {}

  // Member data.
  src_map_type	src_map_;
  dst_map_type	dst_map_;
};



/***********************************************************************
  Matrix copy - using Transpose 
***********************************************************************/

template <typename T,
	  typename SrcBlock,
	  typename DstBlock>
struct t_mcopy<T, SrcBlock, DstBlock, Impl_transpose> : Benchmark_base
{
  typedef typename SrcBlock::map_type src_map_type;
  typedef typename DstBlock::map_type dst_map_type;

  char const* what() { return "t_mcopy<T, SrcBlock, DstBlock, Impl_transpose>"; }
  int ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size) { return size*sizeof(T); }
  int wiob_per_point(length_type size) { return size*sizeof(T); }
  int mem_per_point(length_type size)  { return 2*size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    length_type const M = size;
    length_type const N = size;

    Matrix<T, SrcBlock>   A(M, N, T(), src_map_);
    Matrix<T, DstBlock>   Z(M, N,      dst_map_);

    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	A.put(m, n, T(m*N + n));
      }
    
    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<SrcBlock> src_ext(A.block());
      impl::Ext_data<DstBlock> dst_ext(Z.block());
    
      if (src_ext.stride(1) == 1 && dst_ext.stride(0) == 1)
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	  ipp_transpose(
		src_ext.data(), src_ext.stride(0), src_ext.stride(1),
		dst_ext.data(), dst_ext.stride(1), dst_ext.stride(0),
		M, N);
	t1.stop();
      }
      else if (src_ext.stride(0) == 1 && dst_ext.stride(1) == 1)
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	  ipp_transpose(
		src_ext.data(), src_ext.stride(1), src_ext.stride(0),
		dst_ext.data(), dst_ext.stride(0), dst_ext.stride(1),
		N, M);
	t1.stop();
      }
      else assert(0);
    }
    
    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	if (!equal(Z.get(m, n), T(m*N+n)))
	{
	  std::cout << "t_mcopy: ERROR" << std::endl;
	  std::cout << "A = " << A << std::endl;
	  std::cout << "Z = " << Z << std::endl;
	  abort();
	}
      }
    
    time = t1.delta();
  }

  t_mcopy(src_map_type src_map, dst_map_type dst_map)
    : src_map_(src_map),
      dst_map_(dst_map)
    {}

  // Member data.
  src_map_type	src_map_;
  dst_map_type	dst_map_;
};



/***********************************************************************
  Matrix copy - using select 
***********************************************************************/

template <typename T,
	  typename SrcBlock,
	  typename DstBlock>
struct t_mcopy<T, SrcBlock, DstBlock, Impl_select> : Benchmark_base
{
  typedef typename SrcBlock::map_type src_map_type;
  typedef typename DstBlock::map_type dst_map_type;

  char const* what() { return "t_mcopy<T, SrcBlock, DstBlock, Impl_copy>"; }
  int ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size) { return size*sizeof(T); }
  int wiob_per_point(length_type size) { return size*sizeof(T); }
  int mem_per_point(length_type size)  { return 2*size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    length_type const M = size;
    length_type const N = size;

    Matrix<T, SrcBlock>   A(M, N, T(), src_map_);
    Matrix<T, DstBlock>   Z(M, N,      dst_map_);

    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	A.put(m, n, T(m*N + n));
      }
    
    vsip::impl::profile::Timer t1;

    {
      impl::Ext_data<SrcBlock> src_ext(A.block());
      impl::Ext_data<DstBlock> dst_ext(Z.block());
    
      if (src_ext.stride(1) == 1 && dst_ext.stride(0) == 1)
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	  ipp_transpose(
		src_ext.data(), src_ext.stride(0), src_ext.stride(1),
		dst_ext.data(), dst_ext.stride(1), dst_ext.stride(0),
		M, N);
	t1.stop();
      }
      else
      {
	t1.start();
	for (index_type l=0; l<loop; ++l)
	  ipp_copy(src_ext.data(), src_ext.stride(0), src_ext.stride(1),
		   dst_ext.data(), dst_ext.stride(0), dst_ext.stride(1),
		   M, N);
	t1.stop();
      }
    }
    
    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	if (!equal(Z.get(m, n), T(m*N+n)))
	{
	  std::cout << "t_mcopy: ERROR" << std::endl;
	  std::cout << "A = " << A << std::endl;
	  std::cout << "Z = " << Z << std::endl;
	  abort();
	}
      }
    
    time = t1.delta();
  }

  t_mcopy(src_map_type src_map, dst_map_type dst_map)
    : src_map_(src_map),
      dst_map_(dst_map)
    {}

  // Member data.
  src_map_type	src_map_;
  dst_map_type	dst_map_;
};



/***********************************************************************
  Benchmark driver for local copy
***********************************************************************/

template <typename T,
	  typename SrcOrder,
	  typename DstOrder,
	  typename ImplTag>
struct t_mcopy_local : t_mcopy<T,
			       Dense<2, T, SrcOrder, Local_map>,
			       Dense<2, T, DstOrder, Local_map>,
			       ImplTag>
{
  typedef t_mcopy<T,
		  Dense<2, T, SrcOrder, Local_map>,
		  Dense<2, T, DstOrder, Local_map>,
		  ImplTag> base_type;
  t_mcopy_local()
    : base_type(Local_map(), Local_map()) 
  {}
};


template <typename T,
	  typename SrcOrder,
	  typename DstOrder,
	  typename ImplTag>
struct t_mcopy_pb : t_mcopy<T,
			    Plain_block<2, T, SrcOrder, Local_map>,
			    Plain_block<2, T, DstOrder, Local_map>,
			    ImplTag>
{
  typedef t_mcopy<T,
		  Plain_block<2, T, SrcOrder, Local_map>,
		  Plain_block<2, T, DstOrder, Local_map>,
		  ImplTag > base_type;
  t_mcopy_pb()
    : base_type(Local_map(), Local_map()) 
  {}
};



/***********************************************************************
  Benchmark driver for distributed copy
***********************************************************************/

template <typename T,
	  int      SrcDO,
	  int      DstDO,
	  typename ImplTag>
struct t_mcopy_par_helper
{
  typedef Map<Block_dist, Block_dist> map_type;

  typedef typename ITE_Type<SrcDO == 0,
		   As_type<row2_type>, As_type<col2_type> >::type
          src_order_type;
  typedef Dense<2, T, src_order_type, map_type> src_block_type;

  typedef typename ITE_Type<DstDO == 0,
		   As_type<row2_type>, As_type<col2_type> >::type
          dst_order_type;
  typedef Dense<2, T, dst_order_type, map_type> dst_block_type;

  typedef t_mcopy<T, src_block_type, dst_block_type, ImplTag> base_type;
};


template <typename T,
	  int      SrcDO,
	  int      DstDO,
	  typename ImplTag>
struct t_mcopy_par
  : t_mcopy_par_helper<T, SrcDO, DstDO, ImplTag>::base_type 
{
  typedef t_mcopy_par_helper<T, SrcDO, DstDO, ImplTag> helper_type;
  typedef typename helper_type::map_type  map_type;
  typedef typename helper_type::base_type base_type;

  t_mcopy_par(processor_type src_np, processor_type dst_np)
    : base_type(SrcDO == 0 ? map_type(src_np, 1) : map_type(1, src_np),
		DstDO == 0 ? map_type(dst_np, 1) : map_type(1, dst_np))
  {}
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
  typedef row2_type rt;
  typedef col2_type ct;

  switch (what)
  {
  case  1: loop(t_mcopy_local<float, rt, rt, Impl_select>()); break;
  case  2: loop(t_mcopy_local<float, rt, ct, Impl_select>()); break;
  case  3: loop(t_mcopy_local<float, ct, rt, Impl_select>()); break;
  case  4: loop(t_mcopy_local<float, ct, ct, Impl_select>()); break;

  case 11: loop(t_mcopy_local<float, rt, rt, Impl_copy>()); break;
  case 12: loop(t_mcopy_local<float, rt, ct, Impl_copy>()); break;
  case 13: loop(t_mcopy_local<float, ct, rt, Impl_copy>()); break;
  case 14: loop(t_mcopy_local<float, ct, ct, Impl_copy>()); break;

  // case 21: loop(t_mcopy_local<float, rt, rt, Impl_transpose>()); break;
  case 22: loop(t_mcopy_local<float, rt, ct, Impl_transpose>()); break;
  case 23: loop(t_mcopy_local<float, ct, rt, Impl_transpose>()); break;
  // case 24: loop(t_mcopy_local<float, ct, ct, Impl_transpose>()); break;

  case  0:
    std::cout
      << "mcopy -- float matrix copy using IPP \n"
      << "\n"
      << "   -1: rows -> rows   select algorithm\n"
      << "   -2: rows -> cols   select algorithm\n"
      << "   -3: cols -> rows   select algorithm\n"
      << "   -4: cols -> cols   select algorithm\n"
      << " \n"
      << "  -11: rows -> rows   copy algorithm\n"
      << "  -12: rows -> cols   copy algorithm\n"
      << "  -13: cols -> rows   copy algorithm\n"
      << "  -14: cols -> cols   copy algorithm\n"
      << " \n"
      << "  -22: rows -> cols   transpose algorithm\n"
      << "  -23: cols -> rows   transpose algorithm\n"
      << "\n"
      << "  Default parameters:\n"
      << "    -stop 12        // Largest problem size is 2^12 or 4096\n"
      ;

  default:
    return 0;
  }
  return 1;
}
