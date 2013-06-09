//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Test for arbitrary distributed subsets.

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/parallel.hpp>
#include <test.hpp>
#include "util.hpp"

#define VERBOSE 1

using namespace ovxx;
namespace p = ovxx::parallel;

// 1: Standard dispatch

template <typename DstViewT,
	  typename SrcViewT>
void
assign(DstViewT dst, SrcViewT src, integral_constant<int, 1>)
{
  dst = src;
}



// 2: Assignment

template <typename DstViewT,
	  typename SrcViewT>
void
assign(DstViewT dst, SrcViewT src, integral_constant<int, 2>)
{
  using p::choose_par_assign_impl;
  using p::Assignment;

  dimension_type const dim = DstViewT::dim;
  typedef typename DstViewT::block_type dst_block_type;
  typedef typename SrcViewT::block_type src_block_type;

  typedef typename
    choose_par_assign_impl<dim, dst_block_type, src_block_type, false>::type
    par_assign_type;

  Assignment<dim,
    typename DstViewT::block_type,
    typename SrcViewT::block_type,
    par_assign_type> pa(dst, src);

  pa();
}



// 3: par_expr

template <typename DstViewT,
	  typename SrcViewT>
void
assign(DstViewT dst, SrcViewT src, integral_constant<int, 3>)
{
  // The then and else cases are equivalent.
#if 0
  par_expr(dst, src);
#else
  dimension_type const dim = DstViewT::dim;
  typedef typename DstViewT::block_type dst_block_type;
  typedef typename SrcViewT::block_type src_block_type;

  p::Expression<dim, dst_block_type, src_block_type> pe(dst, src);

  pe();
#endif
}



/***********************************************************************
  Utilities
***********************************************************************/

bool
in_domain(Index<1> const&  idx, Domain<1> const& dom)
{
  return   (idx[0] >= dom.first())
      &&  ((idx[0] - dom.first()) % dom.stride() == 0)
      && (((idx[0] - dom.first()) / dom.stride()) < dom.length());
}

bool
in_domain(Index<2> const&  idx, Domain<2> const& dom)
{
  return in_domain(idx[0], dom[0]) && in_domain(idx[1], dom[1]);
}



/***********************************************************************
  Test driver
***********************************************************************/

template <typename T,
	  int      Impl,
	  typename SrcMapT,
	  typename DstMapT>
void
test_src(SrcMapT&         src_map,
	 DstMapT&         dst_map,
	 length_type      rows,
	 length_type      cols,
	 Domain<2> const& src_dom)
{
  typedef Dense<2, T, row2_type, SrcMapT> src_block_type;
  typedef Dense<2, T, row2_type, DstMapT> dst_block_type;
  typedef Matrix<T, src_block_type>       src_view_type;
  typedef Matrix<T, dst_block_type>       dst_view_type;

  src_view_type src(rows, cols, T(-1), src_map);
  dst_view_type dst(src_dom[0].size(), src_dom[1].size(), T(-2), dst_map);
   
  // setup input.
  if (subblock(src) != no_subblock)
  {
    for (index_type lr=0; lr<src.local().size(0); ++lr)
      for (index_type lc=0; lc<src.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(src, 0, lr); 
	index_type gc = global_from_local_index(src, 1, lc); 
	src.local().put(lr, lc, T(gr*cols + gc));
      }
  }

  assign(dst, src(src_dom), integral_constant<int, Impl>());

  // checkout output.
  if (subblock(dst) != no_subblock)
  {
    for (index_type lr=0; lr<dst.local().size(0); ++lr)
      for (index_type lc=0; lc<dst.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(dst, 0, lr); 
	index_type gc = global_from_local_index(dst, 1, lc); 
	index_type xr = src_dom[0].first() + gr * src_dom[0].stride();
	index_type xc = src_dom[1].first() + gc * src_dom[1].stride();

	T exp = T(xr*cols + xc);
#if VERBOSE
	if (!equal(dst.local().get(lr, lc), exp))
	{
	  std::cout << "test_src dst(" << gr << ", " << gc << "; "
		    << lr << ", " << lc << ") = "
		    << dst.local().get(lr, lc)
		    << "  expected " << exp
		    << std::endl;
	  std::cout << "Impl: " << Impl << std::endl;
	  std::cout << "src_dom: " << src_dom << std::endl;
	  std::cout << "src_map: " << typeid(SrcMapT).name() << std::endl;
	  std::cout << "dst_map: " << typeid(DstMapT).name() << std::endl;
	  std::cout << "dst.local():\n" << dst.local() << std::endl;
	  std::cout << "src.local():\n" << src.local() << std::endl;
	}
#endif
	test_assert(equal(dst.local().get(lr, lc), exp));
      }
  }
}



template <typename T,
	  int      Impl,
	  typename SrcMapT,
	  typename DstMapT>
void
test_dst(SrcMapT&         src_map,
	 DstMapT&         dst_map,
	 length_type      rows,
	 length_type      cols,
	 Domain<2> const& dst_dom)
{
  typedef Dense<2, T, row2_type, SrcMapT> src_block_type;
  typedef Dense<2, T, row2_type, DstMapT> dst_block_type;
  typedef Matrix<T, src_block_type>       src_view_type;
  typedef Matrix<T, dst_block_type>       dst_view_type;

  src_view_type src(dst_dom[0].size(), dst_dom[1].size(), T(-1), src_map);
  dst_view_type dst(rows, cols, T(-2), dst_map);
   
  // setup input.
  if (subblock(src) != no_subblock)
  {
    for (index_type lr=0; lr<src.local().size(0); ++lr)
      for (index_type lc=0; lc<src.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(src, 0, lr); 
	index_type gc = global_from_local_index(src, 1, lc); 
	src.local().put(lr, lc, T(gr*cols + gc));
      }
  }

  assign(dst(dst_dom), src, integral_constant<int, Impl>());

  // checkout output.
  if (subblock(dst) != no_subblock)
  {
    for (index_type lr=0; lr<dst.local().size(0); ++lr)
      for (index_type lc=0; lc<dst.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(dst, 0, lr); 
	index_type gc = global_from_local_index(dst, 1, lc); 

	T exp;

	if (in_domain(Index<2>(gr, gc), dst_dom))
	{
	  index_type xr = (gr - dst_dom[0].first()) / dst_dom[0].stride();
	  index_type xc = (gc - dst_dom[1].first()) / dst_dom[1].stride();

	  exp = T(xr*cols + xc);
	}
	else
	  exp = T(-2);

#if VERBOSE
	if (!equal(dst.local().get(lr, lc), exp))
	{
	  std::cout << "test_dst: dst(" << gr << ", " << gc << ") = "
		    << dst.local().get(lr, lc)
		    << "  expected " << exp
		    << std::endl;
	}
#endif
	test_assert(equal(dst.local().get(lr, lc), exp));
      }
  }
}



// 1-Dim with subdomains on source and destination

template <typename T,
	  int      Impl,
	  typename SrcMapT,
	  typename DstMapT>
void
test_src_dst(SrcMapT&         src_map,
	     DstMapT&         dst_map,
	     length_type      size,
	     Domain<1> const& src_dom,
	     Domain<1> const& dst_dom)
{
  typedef Dense<1, T, row1_type, SrcMapT> src_block_type;
  typedef Dense<1, T, row1_type, DstMapT> dst_block_type;
  typedef Vector<T, src_block_type>       src_view_type;
  typedef Vector<T, dst_block_type>       dst_view_type;

  src_view_type src(size, T(-1), src_map);
  dst_view_type dst(size, T(-3), dst_map);

  dst = T(-2);
   
  // setup input.
  if (subblock(src) != no_subblock)
  {
    for (index_type lr=0; lr<src.local().size(0); ++lr)
    {
      index_type gr = global_from_local_index(src, 0, lr); 
      src.local().put(lr, T(gr));
    }
  }

  assign(dst(dst_dom), src(src_dom), integral_constant<int, Impl>());

  // checkout output.
  if (subblock(dst) != no_subblock)
  {
    for (index_type lr=0; lr<dst.local().size(0); ++lr)
    {
      index_type gr = global_from_local_index(dst, 0, lr); 

      T exp;

      if (in_domain(Index<1>(gr), dst_dom))
      {
	index_type xr = (gr - dst_dom[0].first()) / dst_dom[0].stride();
	
	xr = xr * src_dom[0].stride() + src_dom[0].first();
	
	exp = T(xr);
      }
      else
	exp = T(-2);

#if VERBOSE
      if (!equal(dst.local().get(lr), exp))
      {
	std::cout << "test_src_dst dst(" << gr << "; "
		  << lr << ") = "
		  << dst.local().get(lr)
		  << "  expected " << exp
		  << std::endl;
      }
#endif
      test_assert(equal(dst.local().get(lr), exp));
    }
  }
}



// 2-Dim with subdomains on source and destination

template <typename T,
	  int      Impl,
	  typename SrcMapT,
	  typename DstMapT>
void
test_src_dst(SrcMapT&         src_map,
	     DstMapT&         dst_map,
	     length_type      rows,
	     length_type      cols,
	     Domain<2> const& src_dom,
	     Domain<2> const& dst_dom)
{
  typedef Dense<2, T, row2_type, SrcMapT> src_block_type;
  typedef Dense<2, T, row2_type, DstMapT> dst_block_type;
  typedef Matrix<T, src_block_type>       src_view_type;
  typedef Matrix<T, dst_block_type>       dst_view_type;

  src_view_type src(rows, cols, T(-1), src_map);
  dst_view_type dst(rows, cols, T(-3), dst_map);

  dst = T(-2);
   
  // setup input.
  if (subblock(src) != no_subblock)
  {
    for (index_type lr=0; lr<src.local().size(0); ++lr)
      for (index_type lc=0; lc<src.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(src, 0, lr); 
	index_type gc = global_from_local_index(src, 1, lc); 
	src.local().put(lr, lc, T(gr*cols + gc));
      }
  }

  assign(dst(dst_dom), src(src_dom), integral_constant<int, Impl>());

  // checkout output.
  if (subblock(dst) != no_subblock)
  {
    for (index_type lr=0; lr<dst.local().size(0); ++lr)
      for (index_type lc=0; lc<dst.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(dst, 0, lr); 
	index_type gc = global_from_local_index(dst, 1, lc); 

	T exp;

	if (in_domain(Index<2>(gr, gc), dst_dom))
	{
	  index_type xr = (gr - dst_dom[0].first()) / dst_dom[0].stride();
	  index_type xc = (gc - dst_dom[1].first()) / dst_dom[1].stride();

	  xr = xr * src_dom[0].stride() + src_dom[0].first();
	  xc = xc * src_dom[1].stride() + src_dom[1].first();

	  exp = T(xr*cols + xc);
	}
	else
	  exp = T(-2);

#if VERBOSE
	if (!equal(dst.local().get(lr, lc), exp))
	{
	  std::cout << "test_src_dst dst(" << gr << ", " << gc << "; "
		    << lr << ", " << lc << ") = "
		    << dst.local().get(lr, lc)
		    << "  expected " << exp
		    << std::endl;
	}
#endif
	test_assert(equal(dst.local().get(lr, lc), exp));
      }
  }
}



struct SrcTag {};
struct DstTag {};

template <typename T,
	  typename SrcMapT,
	  typename DstMapT>
void
test_type(SrcTag,
	  SrcMapT&         src_map,
	  DstMapT&         dst_map,
	  length_type      rows,
	  length_type      cols,
	  Domain<2> const& src_dom)
{
  test_src<T, 1>(src_map, dst_map, rows, cols, src_dom);
  test_src<T, 2>(src_map, dst_map, rows, cols, src_dom);
  test_src<T, 3>(src_map, dst_map, rows, cols, src_dom);
}



template <typename T,
	  typename SrcMapT,
	  typename DstMapT>
void
test_type(DstTag,
	  SrcMapT&         src_map,
	  DstMapT&         dst_map,
	  length_type      rows,
	  length_type      cols,
	  Domain<2> const& dst_dom)
{
  test_dst<T, 1>(src_map, dst_map, rows, cols, dst_dom);
  test_dst<T, 2>(src_map, dst_map, rows, cols, dst_dom);
  test_dst<T, 3>(src_map, dst_map, rows, cols, dst_dom);
}



template <typename T,
	  typename Tag>
void
test_map_combinations()
{
  length_type np, nr, nc;
  get_np_square(np, nr, nc);

  Map<>         root(1, 1);
  Map<>         r_map(np, 1);
  Map<>         c_map(1, np);
  Map<>         x_map(nr, nc);
  Replicated_map<2> g_map;

  // coverage tests
  length_type const n_dom = 8;
  Domain<2> dom[n_dom];

  dom[0] = Domain<2>(Domain<1>(0,  1, 128), Domain<1>(0, 1, 64));
  dom[1] = Domain<2>(Domain<1>(64, 1, 128), Domain<1>(32, 1, 64));
  dom[2] = Domain<2>(Domain<1>(0,  1, 32),  Domain<1>(0,  1, 16));
  dom[3] = Domain<2>(Domain<1>(64, 1, 32),  Domain<1>(32, 1, 16));
  dom[4] = Domain<2>(Domain<1>(0,  2, 32),  Domain<1>(0,  2, 16));
  dom[5] = Domain<2>(Domain<1>(64, 2, 32),  Domain<1>(32, 2, 16));
  dom[6] = Domain<2>(Domain<1>(0,  1, 32),  Domain<1>(0,  2, 16));
  dom[7] = Domain<2>(Domain<1>(64, 2, 32),  Domain<1>(32, 1, 16));

  for (index_type i=0; i<n_dom; ++i)
  {
    test_type<float>(Tag(), root,   root, 256, 128, dom[i]);
    test_type<float>(Tag(), r_map,  root, 256, 128, dom[i]);
    test_type<float>(Tag(), c_map,  root, 256, 128, dom[i]);
    test_type<float>(Tag(), x_map,  root, 256, 128, dom[i]);
    test_type<float>(Tag(), g_map,  root, 256, 128, dom[i]);
    test_type<float>(Tag(), root,  r_map, 256, 128, dom[i]);
    test_type<float>(Tag(), r_map, r_map, 256, 128, dom[i]);
    test_type<float>(Tag(), c_map, r_map, 256, 128, dom[i]);
    test_type<float>(Tag(), x_map, r_map, 256, 128, dom[i]);
    test_type<float>(Tag(), g_map, r_map, 256, 128, dom[i]);
    test_type<float>(Tag(), root,  c_map, 256, 128, dom[i]);
    test_type<float>(Tag(), r_map, c_map, 256, 128, dom[i]);
    test_type<float>(Tag(), c_map, c_map, 256, 128, dom[i]);
    test_type<float>(Tag(), x_map, c_map, 256, 128, dom[i]);
    test_type<float>(Tag(), g_map, c_map, 256, 128, dom[i]);
    test_type<float>(Tag(), root,  x_map, 256, 128, dom[i]);
    test_type<float>(Tag(), r_map, x_map, 256, 128, dom[i]);
    test_type<float>(Tag(), c_map, x_map, 256, 128, dom[i]);
    test_type<float>(Tag(), x_map, x_map, 256, 128, dom[i]);
    test_type<float>(Tag(), g_map, x_map, 256, 128, dom[i]);
    test_type<float>(Tag(), root,  g_map, 256, 128, dom[i]);
    test_type<float>(Tag(), r_map, g_map, 256, 128, dom[i]);
    test_type<float>(Tag(), c_map, g_map, 256, 128, dom[i]);
    test_type<float>(Tag(), x_map, g_map, 256, 128, dom[i]);
    test_type<float>(Tag(), g_map, g_map, 256, 128, dom[i]);
  }
}



template <typename T,
	  typename SrcMapT,
	  typename DstMapT>
void
test_src_dst_type(SrcMapT&         src_map,
		  DstMapT&         dst_map,
		  length_type      rows,
		  length_type      cols,
		  Domain<2> const& src_dom,
		  Domain<2> const& dst_dom)
{
  test_src_dst<T, 1>(src_map, dst_map, rows, cols, src_dom, dst_dom);
  test_src_dst<T, 2>(src_map, dst_map, rows, cols, src_dom, dst_dom);
  test_src_dst<T, 3>(src_map, dst_map, rows, cols, src_dom, dst_dom);

  if (src_map.num_subblocks(1) == 1 && dst_map.num_subblocks(1) == 1)
  {
    test_src_dst<T, 1>(src_map, dst_map, rows, src_dom[0], dst_dom[0]);
    test_src_dst<T, 2>(src_map, dst_map, rows, src_dom[0], dst_dom[0]);
    test_src_dst<T, 3>(src_map, dst_map, rows, src_dom[0], dst_dom[0]);
  }
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if 0
  // Enable this section for easier debugging.
  p::Communicator& comm = p::default_communicator();
  pid_t pid = getpid();

  cout << "rank: "   << comm.rank()
       << "  size: " << comm.size()
       << "  pid: "  << pid
       << endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  cout << "start\n";
#endif

  length_type np, nr, nc;
  get_np_square(np, nr, nc);

  Map<> root(1, 1);
  Map<> r_map(np, 1);
  Map<> c_map(1, np);
  Map<> x_map(nr, nc);


#if VSIP_IMPL_TEST_LEVEL >= 0
  // examples of simple testcases -- easier to debug
  test_src<float, 2>(root,  r_map, 8, 8,
		     Domain<2>(Domain<1>(0, 2, 4), Domain<1>(0, 2, 4)));
  test_src<float, 2>(root,  r_map, 8, 8,
		     Domain<2>(Domain<1>(0, 1, 4), Domain<1>(0, 2, 4)));
  test_src<float, 2>(root,  r_map, 8, 8,
		     Domain<2>(Domain<1>(0, 1, 4), Domain<1>(0, 1, 4)));
#endif

#if VSIP_IMPL_TEST_LEVEL >= 2
  test_map_combinations<float, SrcTag>();
  test_map_combinations<float, DstTag>();

  // coverage tests
  length_type const n_dom = 8;
  Domain<2> dom[n_dom];

  dom[0] = Domain<2>(Domain<1>(0,  1, 128), Domain<1>(0, 1, 64));
  dom[1] = Domain<2>(Domain<1>(64, 1, 128), Domain<1>(32, 1, 64));
  dom[2] = Domain<2>(Domain<1>(0,  1, 32),  Domain<1>(0,  1, 16));
  dom[3] = Domain<2>(Domain<1>(64, 1, 32),  Domain<1>(32, 1, 16));
  dom[4] = Domain<2>(Domain<1>(0,  2, 32),  Domain<1>(0,  2, 16));
  dom[5] = Domain<2>(Domain<1>(64, 2, 32),  Domain<1>(32, 2, 16));
  dom[6] = Domain<2>(Domain<1>(0,  1, 32),  Domain<1>(0,  2, 16));
  dom[7] = Domain<2>(Domain<1>(64, 2, 32),  Domain<1>(32, 1, 16));

  for (index_type i=0; i<n_dom; i+=2)
  {
    test_src_dst_type<float>(root,   root, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(r_map,  root, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(c_map,  root, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(x_map,  root, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(root,  r_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(r_map, r_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(c_map, r_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(x_map, r_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(root,  c_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(r_map, c_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(c_map, c_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(x_map, c_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(root,  x_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(r_map, x_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(c_map, x_map, 256, 128, dom[i], dom[i+1]);
    test_src_dst_type<float>(x_map, x_map, 256, 128, dom[i], dom[i+1]);
  }
#endif
}
