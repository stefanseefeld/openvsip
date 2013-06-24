//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for matrix copy (including transpose).

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/map.hpp>
#include "benchmark.hpp"
#include <ovxx/output.hpp>
#include <iostream>
#include <cstring>

using namespace vsip;


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

struct Impl_assign;
struct Impl_sa;
struct Impl_memcpy;

template <typename T,
	  typename SrcBlock,
	  typename DstBlock,
	  typename ImplTag>
struct t_mcopy;



/***********************************************************************
  Matrix copy - normal assignment
***********************************************************************/

template <typename T,
	  typename SrcBlock,
	  typename DstBlock>
struct t_mcopy<T, SrcBlock, DstBlock, Impl_assign> : Benchmark_base
{
  typedef typename SrcBlock::map_type src_map_type;
  typedef typename DstBlock::map_type dst_map_type;

  char const* what() { return "t_mcopy<T, SrcBlock, DstBlock, Impl_assign>"; }
  int ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size) { return size*sizeof(T); }
  int wiob_per_point(length_type size) { return size*sizeof(T); }
  int mem_per_point(length_type size)  { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
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
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      Z = A;
    time = t1.elapsed();
    
    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	if (!equal(Z.get(m, n), T(m*N+n)))
	{
	  std::cout << A << std::endl;
	  std::cout << Z << std::endl;
	  std::cout << "t_mcopy: ERROR" << std::endl;
	  abort();
	}
      }
  }

  void diag()
  {
    using namespace ovxx;

    length_type const M = 256;
    length_type const N = 256;

    Matrix<T, SrcBlock>   A(M, N, T(), src_map_);
    Matrix<T, DstBlock>   Z(M, N,      dst_map_);

    std::cout << assignment::diagnostics(Z, A) << std::endl;
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
  Matrix copy - using memcpy 
***********************************************************************/

template <typename T,
	  typename SrcBlock,
	  typename DstBlock>
struct t_mcopy<T, SrcBlock, DstBlock, Impl_memcpy> : Benchmark_base
{
  typedef typename SrcBlock::map_type src_map_type;
  typedef typename DstBlock::map_type dst_map_type;

  char const* what() { return "t_mcopy<T, SrcBlock, DstBlock, Impl_memcpy>"; }
  int ops_per_point(length_type size)  { return size; }
  int riob_per_point(length_type size) { return size*sizeof(T); }
  int wiob_per_point(length_type size) { return size*sizeof(T); }
  int mem_per_point(length_type size)  { return size*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time) OVXX_NOINLINE
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
    
    {
      dda::Data<SrcBlock, dda::in> src_data(A.block());
      dda::Data<DstBlock, dda::out> dst_data(Z.block());
    
      timer t1;
      for (index_type l=0; l<loop; ++l)
	std::memcpy(dst_data.ptr(), src_data.ptr(), M*N*sizeof(T));
      time = t1.elapsed();
    }
    
    for (index_type m=0; m<M; ++m)
      for (index_type n=0; n<N; ++n)
      {
	if (!equal(Z.get(m, n), T(m*N+n)))
	{
	  std::cout << "t_mcopy: ERROR" << std::endl;
	  abort();
	}
      }
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

  typedef typename ovxx::conditional<SrcDO == 0, row2_type, col2_type>::type
          src_order_type;
  typedef Dense<2, T, src_order_type, map_type> src_block_type;

  typedef typename ovxx::conditional<DstDO == 0, row2_type, col2_type>::type
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
benchmark(Loop1P& loop, int what)
{
  processor_type np = num_processors();

  typedef row2_type rt;
  typedef col2_type ct;

  switch (what)
  {
  case  1: loop(t_mcopy_local<float, rt, rt, Impl_assign>()); break;
  case  2: loop(t_mcopy_local<float, rt, ct, Impl_assign>()); break;
  case  3: loop(t_mcopy_local<float, ct, rt, Impl_assign>()); break;
  case  4: loop(t_mcopy_local<float, ct, ct, Impl_assign>()); break;

  case  5: loop(t_mcopy_local<complex<float>, rt, rt, Impl_assign>()); break;
  case  6: loop(t_mcopy_local<complex<float>, rt, ct, Impl_assign>()); break;
  case  7: loop(t_mcopy_local<complex<float>, ct, rt, Impl_assign>()); break;
  case  8: loop(t_mcopy_local<complex<float>, ct, ct, Impl_assign>()); break;

  case  11: loop(t_mcopy_local<float, rt, rt, Impl_memcpy>()); break;
  // case  12: loop(t_mcopy_local<float, rt, ct, Impl_memcpy>()); break;
  // case  13: loop(t_mcopy_local<float, ct, rt, Impl_memcpy>()); break;
  case  14: loop(t_mcopy_local<float, ct, ct, Impl_memcpy>()); break;
    
#if OVXX_PARALLEL_API == 1
  case  31: loop(t_mcopy_par<float, 0, 0, Impl_assign>(np, np)); break;
  case  32: loop(t_mcopy_par<float, 0, 1, Impl_assign>(np, np)); break;

#endif
  case 102: loop(t_mcopy_local<int, rt, ct, Impl_assign>()); break;

  case   0:
    std::cout
      << "mcopy -- matrix copy with and without transpose\n"
      << "    -1: local,         float,  rows <- rows, assignment\n"
      << "    -2: local,         float,  rows <- cols, assignment\n"
      << "    -3: local,         float,  cols <- rows, assignment\n"
      << "    -4: local,         float,  cols <- cols, assignment\n"
      << "    -5: local, complex<float>, rows <- rows, assignment\n"
      << "    -6: local, complex<float>, rows <- cols, assignment\n"
      << "    -7: local, complex<float>, cols <- rows, assignment\n"
      << "    -8: local, complex<float>, cols <- cols, assignment\n"
      << "   -11: local,         float,  rows <- rows, memcpy\n"
      << "   -14: local,         float,  cols <- cols, memcpy\n"
      << "   -31:   par,         float,  Map<>(np,1) <- Map<>(np,1), assignment\n"
      << "   -32:   par,         float,  Map<>(np,1) <- Map<>(1,np), assignment\n"
      << "  -102: local,           int,  rows <- cols, assignment\n"
      << "\n"
      << " Notes:\n"
      << "   par -- distributed blocks\n"
      ;

  default:
    return 0;
  }
  return 1;
}
