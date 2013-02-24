/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Unit tests for distributed subviews.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/math.hpp>
#include <vsip/tensor.hpp>
#include <vsip/parallel.hpp>

#include <vsip/opt/profile.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "util.hpp"
#include "util-par.hpp"

using namespace vsip;

/***********************************************************************
  Test row-subviews of distributed matrix
***********************************************************************/


template <typename T,
	  typename MapT>
void
test_row_sum(
  Domain<2> const& dom,
  MapT const&      map)
{
  typedef Map<>                              root_map_t;
  typedef Dense<1, T, row1_type, root_map_t> root_block1_t;
  typedef Dense<2, T, row2_type, root_map_t> root_block2_t;
  typedef Vector<T, root_block1_t>           root_view1_t;
  typedef Matrix<T, root_block2_t>           root_view2_t;

  typedef Dense<2, T, row2_type, MapT> block2_t;
  typedef Matrix<T, block2_t>          view2_t;

  typedef Vector<T, Dense<1, T, row1_type, Replicated_map<1> > > replicated_view1_t;

  impl::Communicator& comm = impl::default_communicator();

  length_type sum_size = dom[1].size();

  bool const verbose = false;

  // ------------------------------------------------------------------ 
  // Initialize matrix on root processor

  root_map_t   root_map(Block_dist(1),Block_dist(1),Block_dist(1));
  root_view2_t root_view(create_view<root_view2_t>(dom, root_map));
  root_view1_t root_sum (sum_size, T(), root_map);

  if (root_map.subblock() != no_subblock)
  {
    typename root_view2_t::local_type local_view = root_view.local();
    typename root_view1_t::local_type local_sum  = root_sum.local();

    for (index_type r=0; r<local_view.size(0); ++r)
      for (index_type c=0; c<local_view.size(1); ++c)
	local_view(r, c) = T(1000*r + c);

    for (index_type r=0; r<local_view.size(0); ++r)
      local_sum += local_view.row(r);
  }


  // ------------------------------------------------------------------ 
  // Distribute matrix and answer

  view2_t view(create_view<view2_t>(dom, map));
  view = root_view;

  replicated_view1_t chk_sum(sum_size);
  chk_sum = root_sum;


  // ------------------------------------------------------------------ 
  // Sum view into replicated sum

  replicated_view1_t my_sum(dom[1].size(), T());

  if (verbose) dump_map<2>(view.block().map());
  for (index_type r=0; r<view.size(0); ++r)
  {
    if (verbose)
    {
      comm.barrier();
      if (comm.rank() == 0)
      {
	std::cout << "---------------------------------------\n";
	std::cout << "row " << r << std::endl;
	std::cout << "---------------------------------------\n";
      }
      comm.barrier();
      
      dump_map<1>(view.row(r).block().map());
      comm.barrier();
    }
    my_sum = my_sum + view.row(r);
  }

  typename replicated_view1_t::local_type local_my_sum  = my_sum.local();
  typename replicated_view1_t::local_type local_chk_sum = chk_sum.local();

  for (index_type i=0; i<sum_size; ++i)
  {
    test_assert(local_my_sum(i) == local_chk_sum(i));
  }

}



template <typename T>
void
cases_row_sum(Domain<2> const& dom)
{
  length_type np, nr, nc;

  get_np_square(np, nr, nc);

  test_row_sum<T>(dom, Map<>(Block_dist(np), Block_dist(1)));
  test_row_sum<T>(dom, Map<>(Block_dist(1),  Block_dist(np)));
  test_row_sum<T>(dom, Map<>(Block_dist(nr), Block_dist(nc)));

  test_row_sum<T>(dom, Replicated_map<2>());
}



/***********************************************************************
  Test column-subviews of distributed matrix
***********************************************************************/

template <typename T,
	  typename MapT>
void
test_col_sum(
  Domain<2> const& dom,
  MapT const&      map)
{
  typedef Map<>                              root_map_t;
  typedef Dense<1, T, row1_type, root_map_t> root_block1_t;
  typedef Dense<2, T, row2_type, root_map_t> root_block2_t;
  typedef Vector<T, root_block1_t>           root_view1_t;
  typedef Matrix<T, root_block2_t>           root_view2_t;

  typedef Dense<2, T, row2_type, MapT> block2_t;
  typedef Matrix<T, block2_t>          view2_t;

  typedef Vector<T, Dense<1, T, row1_type, Replicated_map<1> > > replicated_view1_t;

  impl::Communicator& comm = impl::default_communicator();

  length_type sum_size = dom[0].size();

  bool const verbose = false;

  // ------------------------------------------------------------------ 
  // Initialize matrix on root processor

  root_map_t   root_map(Block_dist(1),Block_dist(1),Block_dist(1));
  root_view2_t root_view(create_view<root_view2_t>(dom, root_map));
  root_view1_t root_sum (sum_size, T(), root_map);

  if (root_map.subblock() != no_subblock)
  {
    typename root_view2_t::local_type local_view = root_view.local();
    typename root_view1_t::local_type local_sum  = root_sum.local();

    for (index_type r=0; r<local_view.size(0); ++r)
      for (index_type c=0; c<local_view.size(1); ++c)
	local_view(r, c) = T(1000*r + c);

    for (index_type c=0; c<local_view.size(1); ++c)
      local_sum += local_view.col(c);
  }


  // ------------------------------------------------------------------ 
  // Distribute matrix and answer

  view2_t view(create_view<view2_t>(dom, map));
  view = root_view;

  replicated_view1_t chk_sum(sum_size);
  chk_sum = root_sum;


  // ------------------------------------------------------------------ 
  // Sum view into replicated sum

  replicated_view1_t my_sum(sum_size, T());

  if (verbose) dump_map<2>(view.block().map());
  for (index_type c=0; c<view.size(1); ++c)
  {
    if (verbose)
    {
      comm.barrier();
      if (comm.rank() == 0)
      {
	std::cout << "---------------------------------------\n";
	std::cout << "col " << c << std::endl;
	std::cout << "---------------------------------------\n";
      }
      comm.barrier();
      
      dump_map<1>(view.col(c).block().map());
      comm.barrier();
    }
    my_sum = my_sum + view.col(c);
  }

  typename replicated_view1_t::local_type local_my_sum  = my_sum.local();
  typename replicated_view1_t::local_type local_chk_sum = chk_sum.local();

  for (index_type i=0; i<sum_size; ++i)
  {
    test_assert(local_my_sum(i) == local_chk_sum(i));
  }

}



template <typename T>
void
cases_col_sum(Domain<2> const& dom)
{
  length_type np, nr, nc;

  get_np_square(np, nr, nc);

  test_col_sum<T>(dom, Map<>(Block_dist(np), Block_dist(1)));
  test_col_sum<T>(dom, Map<>(Block_dist(1),  Block_dist(np)));
  test_col_sum<T>(dom, Map<>(Block_dist(nr), Block_dist(nc)));
}



/***********************************************************************
  Test vector-subviews of distributed tensors
***********************************************************************/

template <dimension_type Slice>
struct V_slice;

template <>
struct V_slice<0>
{
  length_type size0   (Domain<3> const& dom) { return dom[1].size(); }
  length_type size1   (Domain<3> const& dom) { return dom[2].size(); }
  length_type sum_size(Domain<3> const& dom) { return dom[0].size(); }

  template <typename T,
	    typename Block>
  typename Tensor<T, Block>::template subvector<1, 2>::type
  subview(Tensor<T, Block> ten, index_type idx0, index_type idx1)
  { return ten(vsip::whole_domain, idx0, idx1); }
};

template <>
struct V_slice<1>
{
  length_type size0   (Domain<3> const& dom) { return dom[0].size(); }
  length_type size1   (Domain<3> const& dom) { return dom[2].size(); }
  length_type sum_size(Domain<3> const& dom) { return dom[1].size(); }

  template <typename T,
	    typename Block>
  typename Tensor<T, Block>::template subvector<0, 2>::type
  subview(Tensor<T, Block> ten, index_type idx0, index_type idx1)
  { return ten(idx0, vsip::whole_domain, idx1); }
};

template <>
struct V_slice<2>
{
  length_type size0   (Domain<3> const& dom) { return dom[0].size(); }
  length_type size1   (Domain<3> const& dom) { return dom[1].size(); }
  length_type sum_size(Domain<3> const& dom) { return dom[2].size(); }

  template <typename T,
	    typename Block>
  typename Tensor<T, Block>::template subvector<0, 1>::type
  subview(Tensor<T, Block> ten, index_type idx0, index_type idx1)
  { return ten(idx0, idx1, vsip::whole_domain); }
};


template <typename       T,
	  dimension_type Slice,
	  typename       MapT>
void
test_tensor_v_sum(
  Domain<3> const& dom,
  MapT const&      map)
{
  typedef Map<>                              root_map_t;
  typedef Dense<1, T, row1_type, root_map_t> root_block1_t;
  typedef Dense<3, T, row3_type, root_map_t> root_block3_t;
  typedef Vector<T, root_block1_t>           root_view1_t;
  typedef Tensor<T, root_block3_t>           root_view3_t;

  typedef Dense<3, T, row2_type, MapT> block3_t;
  typedef Tensor<T, block3_t>          view3_t;

  typedef Vector<T, Dense<1, T, row1_type, Replicated_map<1> > > replicated_view1_t;

  V_slice<Slice> slice;

  impl::Communicator& comm = impl::default_communicator();

  length_type sum_size = slice.sum_size(dom);

  bool const verbose = false;


  // ------------------------------------------------------------------ 
  // Initialize matrix on root processor

  root_map_t   root_map(Block_dist(1),Block_dist(1),Block_dist(1));
  root_view3_t root_view(create_view<root_view3_t>(dom, root_map));
  root_view1_t root_sum (sum_size, T(), root_map);

  if (root_map.subblock() != no_subblock)
  {
    typename root_view3_t::local_type local_view = root_view.local();
    typename root_view1_t::local_type local_sum  = root_sum.local();

    for (index_type i=0; i<local_view.size(0); ++i)
      for (index_type j=0; j<local_view.size(1); ++j)
	for (index_type k=0; k<local_view.size(2); ++k)
	  local_view(i, j, k) = T(10000*i + 100*j + k);

    for (index_type i=0; i<slice.size0(dom); ++i)
      for (index_type j=0; j<slice.size1(dom); ++j)
	local_sum += slice.subview(local_view, i, j);
  }


  // ------------------------------------------------------------------ 
  // Distribute view and answer

  view3_t view(create_view<view3_t>(dom, map));
  view = root_view;

  replicated_view1_t chk_sum(sum_size);
  chk_sum = root_sum;


  // ------------------------------------------------------------------ 
  // Sum view into replicated sum

  replicated_view1_t my_sum(sum_size, T());

  if (verbose) dump_map<3>(view.block().map());
  for (index_type i=0; i<slice.size0(dom); ++i)
    for (index_type j=0; j<slice.size1(dom); ++j)
    {
      if (verbose)
      {
	comm.barrier();
	if (comm.rank() == 0)
	{
	  std::cout << "---------------------------------------\n";
	  std::cout << "v_slice " << i << ", " << j << std::endl;
	  std::cout << "---------------------------------------\n";
	}
	comm.barrier();

	dump_map<1>(slice.subview(view, i, j).block().map());
	comm.barrier();
      }
      my_sum = my_sum + slice.subview(view, i, j);
  }


  // ------------------------------------------------------------------ 
  // Check answer

  typename replicated_view1_t::local_type local_my_sum  = my_sum.local();
  typename replicated_view1_t::local_type local_chk_sum = chk_sum.local();

  for (index_type i=0; i<sum_size; ++i)
  {
    test_assert(local_my_sum(i) == local_chk_sum(i));
  }
}



template <typename       T,
	  dimension_type Slice>
void
cases_tensor_v_sum(Domain<3> const& dom)
{
  length_type np, nr, nc, nt;

  get_np_square(np, nr, nc);

  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(np), Block_dist(1),  Block_dist(1)));
  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(1),  Block_dist(np), Block_dist(1)));
  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(1),  Block_dist(1),  Block_dist(np)));

  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(nr), Block_dist(nc), Block_dist(1)));
  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(nr), Block_dist(1),  Block_dist(nc)));
  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(1),  Block_dist(nr), Block_dist(nc)));

  get_np_cube(np, nr, nc, nt);

  test_tensor_v_sum<T, Slice>(
    dom, Map<>(Block_dist(nr),  Block_dist(nc), Block_dist(nt)));
}



/***********************************************************************
  Test matrix-subviews of distributed tensors
***********************************************************************/

template <dimension_type Slice>
struct M_slice;

template <>
struct M_slice<0>
{
  length_type slices  (Domain<3> const& dom) { return dom[0].size(); }
  length_type sum_rows(Domain<3> const& dom) { return dom[1].size(); }
  length_type sum_cols(Domain<3> const& dom) { return dom[2].size(); }

  template <typename T,
	    typename Block>
  typename Tensor<T, Block>::template submatrix<0>::type
  subview(Tensor<T, Block> ten, index_type i)
  { return ten(i, vsip::whole_domain, vsip::whole_domain); }
};

template <>
struct M_slice<1>
{
  length_type slices  (Domain<3> const& dom) { return dom[1].size(); }
  length_type sum_rows(Domain<3> const& dom) { return dom[0].size(); }
  length_type sum_cols(Domain<3> const& dom) { return dom[2].size(); }

  template <typename T,
	    typename Block>
  typename Tensor<T, Block>::template submatrix<1>::type
  subview(Tensor<T, Block> ten, index_type i)
    { return ten(vsip::whole_domain, i, vsip::whole_domain); }
};

template <>
struct M_slice<2>
{
  length_type slices  (Domain<3> const& dom) { return dom[2].size(); }
  length_type sum_rows(Domain<3> const& dom) { return dom[0].size(); }
  length_type sum_cols(Domain<3> const& dom) { return dom[1].size(); }

  template <typename T,
	    typename Block>
  typename Tensor<T, Block>::template submatrix<2>::type
  subview(Tensor<T, Block> ten, index_type i)
    { return ten(vsip::whole_domain, vsip::whole_domain, i); }
};



template <typename       T,
	  dimension_type Slice,
	  typename       MapT>
void
test_tensor_m_sum(
  Domain<3> const& dom,
  MapT const&      map)
{
  typedef Map<>                              root_map_t;
  typedef Dense<2, T, row1_type, root_map_t> root_block2_t;
  typedef Dense<3, T, row3_type, root_map_t> root_block3_t;
  typedef Matrix<T, root_block2_t>           root_view2_t;
  typedef Tensor<T, root_block3_t>           root_view3_t;

  typedef Dense<3, T, row2_type, MapT> block3_t;
  typedef Tensor<T, block3_t>          view3_t;

  typedef Matrix<T, Dense<2, T, row1_type, Replicated_map<2> > > replicated_view2_t;

  M_slice<Slice> slice;

  impl::Communicator& comm = impl::default_communicator();

  length_type sum_rows = slice.sum_rows(dom);
  length_type sum_cols = slice.sum_cols(dom);

  bool const verbose = false;

  // ------------------------------------------------------------------ 
  // Initialize matrix on root processor

  root_map_t   root_map(Block_dist(1),Block_dist(1),Block_dist(1));
  root_view3_t root_view(create_view<root_view3_t>(dom, root_map));
  root_view2_t root_sum (sum_rows, sum_cols, T(), root_map);

  if (root_map.subblock() != no_subblock)
  {
    typename root_view3_t::local_type local_view = root_view.local();
    typename root_view2_t::local_type local_sum  = root_sum.local();

    for (index_type i=0; i<local_view.size(0); ++i)
      for (index_type j=0; j<local_view.size(1); ++j)
	for (index_type k=0; k<local_view.size(2); ++k)
	  local_view(i, j, k) = T(10000*i + 100*j + k);

    for (index_type i=0; i<slice.slices(dom); ++i)
      local_sum += slice.subview(local_view, i);
  }


  // ------------------------------------------------------------------ 
  // Distribute view and answer

  view3_t view(create_view<view3_t>(dom, map));
  view = root_view;

  replicated_view2_t chk_sum(sum_rows, sum_cols);
  chk_sum = root_sum;


  // ------------------------------------------------------------------ 
  // Sum view into replicated sum

  replicated_view2_t my_sum(sum_rows, sum_cols, T());

  if (verbose) dump_map<3>(view.block().map());
  for (index_type i=0; i<slice.slices(dom); ++i)
  {
    if (verbose)
    {
      comm.barrier();
      if (comm.rank() == 0)
      {
	std::cout << "---------------------------------------\n";
	std::cout << "slice " << i << std::endl;
	std::cout << "---------------------------------------\n";
      }
      comm.barrier();
      
      dump_map<2>(slice.subview(view, i).block().map());
      comm.barrier();
    }
    my_sum = my_sum + slice.subview(view, i);
  }

  typename replicated_view2_t::local_type local_my_sum  = my_sum.local();
  typename replicated_view2_t::local_type local_chk_sum = chk_sum.local();

  for (index_type i=0; i<sum_rows; ++i)
    for (index_type j=0; j<sum_cols; ++j)
    {
      test_assert(local_my_sum(i, j) == local_chk_sum(i, j));
    }
}



template <typename       T,
	  dimension_type Slice>
void
cases_tensor_m_sum(Domain<3> const& dom)
{
  length_type np, nr, nc, nt;

  get_np_square(np, nr, nc);

  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(np), Block_dist(1),  Block_dist(1)));
  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(1),  Block_dist(np), Block_dist(1)));
  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(1),  Block_dist(1),  Block_dist(np)));

  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(nr), Block_dist(nc), Block_dist(1)));
  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(nr), Block_dist(1),  Block_dist(nc)));
  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(1),  Block_dist(nr), Block_dist(nc)));

  get_np_cube(np, nr, nc, nt);

  test_tensor_m_sum<T, Slice>(
    dom, Map<>(Block_dist(nr),  Block_dist(nc), Block_dist(nt)));
}



int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  bool do_mrow = false;
  bool do_mcol = false;
  bool do_tmat = false;
  bool do_tvec = false;
  bool do_all  = false;
  bool verbose = false;

  int cnt = 0;

  for (int i=1; i<argc; ++i)
  {
    if      (!strcmp(argv[i], "-mrow")) { cnt++; do_mrow = true; }
    else if (!strcmp(argv[i], "-mcol")) { cnt++; do_mcol = true; }
    else if (!strcmp(argv[i], "-tmat")) { cnt++; do_tmat = true; }
    else if (!strcmp(argv[i], "-tvec")) { cnt++; do_tvec = true; }
    else if (!strcmp(argv[i], "-none")) { cnt++; }
  }

  if (cnt == 0)
    do_all = true;

  // int loop = argc > 1 ? atoi(argv[1]) : 1;

#if 0
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
  pid_t pid = getpid();

  std::cout << "rank: "   << comm.rank()
	    << "  size: " << comm.size()
	    << "  pid: "  << pid
	    << std::endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
#endif

  length_type np, nr, nc;

  get_np_square(np, nr, nc);

  if (do_all || do_mrow)
  {
    if (verbose) std::cout <<  "mrow" << std::endl;
    // cases_row_sum<float>          (Domain<2>(4, 15));
    cases_row_sum<complex<float> >(Domain<2>(4, 5));
  }

  if (do_all || do_mcol)
  {
    if (verbose) std::cout <<  "mcol" << std::endl;
    cases_col_sum<float>          (Domain<2>(15, 4));
    cases_col_sum<complex<float> >(Domain<2>(3, 4));
  }

  if (do_all || do_tmat)
  {
    if (verbose) std::cout <<  "tmat" << std::endl;
    // small cases
    cases_tensor_m_sum<float,          0>(Domain<3>(4, 6, 8));
    cases_tensor_m_sum<complex<float>, 0>(Domain<3>(4, 3, 2));
#if VSIP_IMPL_TEST_LEVEL >= 2
    cases_tensor_m_sum<float, 0>(Domain<3>(6, 4, 8));
    cases_tensor_m_sum<float, 0>(Domain<3>(8, 6, 4));
    cases_tensor_m_sum<float, 1>(Domain<3>(4, 6, 8));
    cases_tensor_m_sum<float, 1>(Domain<3>(6, 4, 8));
    cases_tensor_m_sum<float, 1>(Domain<3>(8, 6, 4));
    cases_tensor_m_sum<float, 2>(Domain<3>(4, 6, 8));
    cases_tensor_m_sum<float, 2>(Domain<3>(6, 4, 8));
    cases_tensor_m_sum<float, 2>(Domain<3>(8, 6, 4));
#endif // VSIP_IMPL_TEST_LEVEL >= 2
  }

  if (do_all || do_tvec)
  {
    if (verbose) std::cout <<  "tvec" << std::endl;
    cases_tensor_v_sum<float,          0>(Domain<3>(4, 6, 8));
    cases_tensor_v_sum<complex<float>, 0>(Domain<3>(4, 3, 2));
#if VSIP_IMPL_TEST_LEVEL >= 2
    cases_tensor_v_sum<float, 1>(Domain<3>(4, 6, 8));
    cases_tensor_v_sum<float, 2>(Domain<3>(4, 6, 8));

    cases_tensor_v_sum<float, 0>(Domain<3>(32, 16, 64));
    cases_tensor_v_sum<float, 1>(Domain<3>(32, 16, 64));
    cases_tensor_v_sum<float, 2>(Domain<3>(32, 16, 64));
#endif // VSIP_IMPL_TEST_LEVEL >= 2
  }

  return 0;
}
