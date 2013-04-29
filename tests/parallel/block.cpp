//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define TEST_OLD_PAR_ASSIGN 0

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/tensor.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>

#if TEST_OLD_PAR_ASSIGN
#include <vsip/impl/par-assign.hpp>
#endif

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "util.hpp"
#include "util-par.hpp"

#define IMPL_TAG impl::par_assign_impl_type

using namespace std;
using namespace vsip;

using vsip::impl::Length;
using vsip::impl::extent;

/***********************************************************************
  Definitions
***********************************************************************/

template <typename       T,
	  dimension_type Dim,
	  typename       MapT>
void
test_local_view(
  Domain<Dim> dom,
  MapT        map)
{
  typedef typename impl::Row_major<Dim>::type              order_t;
  typedef Dense<Dim, T, order_t, MapT>                     dist_block_t;
  typedef typename impl::view_of<dist_block_t>::type view_t;

  view_t view(create_view<view_t>(dom, T(), map));

  check_local_view<Dim>(view);
}



// Test a distributed view w/explicit parallel assign.

// Requires:
//   DOM is the extent of the distributed view (1 or 2 dimensional),
//   MAP1 is a distributed mapping,
//   MAP2 is a distributed mapping.
//
// Description:
//  - Initialize view0 to 'identity', each element is a linear
//    function of its global index.  View0 is mapped to processor 0.
//  - Scatter: Perform parallel assignment: view1 = view0, where view1
//    has a parallel mapping.
//  - Distributed increment by 100.
//  - Corner-turn: Perform parallel assignment: view2 = view1, where
//    view2 has a parallel mapping.
//  - Distributed increment by 100.
//  - Gather: Perform parallel assignment: view0 = view2, to bring
//    data back onto processor 0.
//  - Check data in view0.

struct TestImplicit;
template <typename ParAssignTag> struct TestExplicit;

template <typename       TestImplTag,
	  typename       T,
	  dimension_type Dim,
	  typename       Map1,
	  typename       Map2>
struct Test_distributed_view;

template <typename       ParAssignTag,
	  typename       T,
	  dimension_type Dim,
	  typename       Map1,
	  typename       Map2>
struct Test_distributed_view<TestExplicit<ParAssignTag>,
			     T, Dim, Map1, Map2>
{
  static void
  exec(
    Domain<Dim> dom,
    Map1        map1,
    Map2        map2,
    int         loop)
  {
    typedef Map<Block_dist, Block_dist> map0_t;
    
    typedef typename impl::Row_major<Dim>::type order_type;
  
    typedef Dense<Dim, T, order_type, map0_t> dist_block0_t;
    typedef Dense<Dim, T, order_type, Map1>   dist_block1_t;
    typedef Dense<Dim, T, order_type, Map2>   dist_block2_t;

    typedef typename impl::view_of<dist_block0_t>::type view0_t;
    typedef typename impl::view_of<dist_block1_t>::type view1_t;
    typedef typename impl::view_of<dist_block2_t>::type view2_t;

    // map0 is not distributed (effectively).
    map0_t  map0(Block_dist(1), Block_dist(1));

    view0_t view0(create_view<view0_t>(dom, T(), map0));
    view1_t view1(create_view<view1_t>(dom, map1));
    view2_t view2(create_view<view2_t>(dom, map2));

    check_local_view<Dim>(view0);
    check_local_view<Dim>(view1);
    check_local_view<Dim>(view2);

    impl::Communicator& comm = impl::default_communicator();

    // Declare assignments, allows early binding to be done.
    impl::Par_assign<Dim, T, T, dist_block1_t, dist_block0_t, ParAssignTag>
		a1(view1, view0);
    impl::Par_assign<Dim, T, T, dist_block2_t, dist_block1_t, ParAssignTag>
		a2(view2, view1);
    impl::Par_assign<Dim, T, T, dist_block0_t, dist_block2_t, ParAssignTag>
		a3(view0, view2);

    for (int l=0; l<loop; ++l)
    {
      foreach_point(view0, Set_identity<Dim>(dom));

      a1(); // view1 = view0;

      foreach_point(view1, Increment<Dim, T>(T(100)));

      a2(); // view2 = view1;
    
      foreach_point(view2, Increment<Dim, T>(T(1000)));
    
      a3(); // view0 = view2;
    }

    // Check results.
    comm.barrier();

    typename view0_t::local_type local_view = view0.local();

    if (local_processor() == 0) 
    {
      // On processor 0, local_view should be entire view.
      test_assert(extent(local_view) == extent(dom));

      // Check that each value is correct.
      bool good = true;
      Length<Dim> ext = extent(local_view);
      for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
      {
	T expected_value = T();
	for (dimension_type d=0; d<Dim; ++d)
	{
	  expected_value *= local_view.size(d);
	  expected_value += idx[d];
	}
	expected_value += T(1100);

	if (get(local_view, idx) != expected_value)
	{
	  cout << "FAIL: index: " << idx
	       << "  expected " << expected_value
	       << "  got "      << get(local_view, idx)
	       << endl;
	  good = false;
	}
      }

      // cout << "CHECK: " << (good ? "good" : "BAD") << endl;
      test_assert(good);
    }
    else
    {
      // Otherwise, local_view should be empty:
      test_assert(local_view.size() == 0);
    }
  }
};



// Test a distributed view w/implicit parallel assign through assignment op.

template <typename       T,
	  dimension_type Dim,
	  typename       Map1,
	  typename       Map2>
struct Test_distributed_view<TestImplicit, T, Dim, Map1, Map2>
{
  static void
  exec(
    Domain<Dim> dom,
    Map1        map1,
    Map2        map2,
    int         loop)
  {
    typedef Map<Block_dist, Block_dist> map0_t;

    typedef typename impl::Row_major<Dim>::type order_type;
    
    typedef Dense<Dim, T, order_type, map0_t> dist_block0_t;
    typedef Dense<Dim, T, order_type, Map1>   dist_block1_t;
    typedef Dense<Dim, T, order_type, Map2>   dist_block2_t;
    
    typedef typename impl::view_of<dist_block0_t>::type view0_t;
    typedef typename impl::view_of<dist_block1_t>::type view1_t;
    typedef typename impl::view_of<dist_block2_t>::type view2_t;
    
    // map0 is not distributed (effectively).
    map0_t  map0(Block_dist(1), Block_dist(1));
    
    view0_t view0(create_view<view0_t>(dom, T(), map0));
    view1_t view1(create_view<view1_t>(dom, map1));
    view2_t view2(create_view<view2_t>(dom, map2));
    
    check_local_view<Dim>(view0);
    check_local_view<Dim>(view1);
    check_local_view<Dim>(view2);

    impl::Communicator& comm = impl::default_communicator();

    // cout << "(" << local_processor() << "): test_distributed_view\n";

    for (int l=0; l<loop; ++l)
    {
      foreach_point(view0, Set_identity<Dim>(dom));
      
      view1 = view0;
      
      foreach_point(view1, Increment<Dim, T>(T(100)));
      
      view2 = view1;
      
      foreach_point(view2, Increment<Dim, T>(T(1000)));
      
      view0 = view2;
    }
    
    // Check results.
    comm.barrier();

    if (local_processor() == 0) 
    {
      typename view0_t::local_type local_view = view0.local();

      // Check that local_view is in fact the entire view.
      test_assert(extent(local_view) == extent(dom));
      
      // Check that each value is correct.
      bool good = true;
      Length<Dim> ext = extent(local_view);
      for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
      {
	T expected_value = T();
	for (dimension_type d=0; d<Dim; ++d)
	{
	  expected_value *= local_view.size(d);
	  expected_value += idx[d];
	}
	expected_value += T(1100);

	if (get(local_view, idx) != expected_value)
	{
	  cout << "FAIL: index: " << idx
	       << "  expected " << expected_value
	       << "  got "      << get(local_view, idx)
	       << endl;
	  good = false;
	}
      }

      // cout << "CHECK: " << (good ? "good" : "BAD") << endl;
      test_assert(good);
    }
  }
};



// Wrapper for Test_distributed_view

template <typename       TestImplTag,
	  typename       T,
	  dimension_type Dim,
	  typename       Map1,
	  typename       Map2>
void
test_distributed_view(
  Domain<Dim> dom,
  Map1        map1,
  Map2        map2,
  int         loop)
{
  Test_distributed_view<TestImplTag, T, Dim, Map1, Map2>
    ::exec(dom, map1, map2, loop);
}



// Test several distributed vector cases for a given type and parallel
// assignment implementation.

template <typename TestImplTag,
	  typename T>
void
test_vector(int loop)
{
  processor_type np = num_processors();

  test_distributed_view<TestImplTag, T>(
    Domain<1>(16),
    Map<Block_dist>(np),
    Map<Block_dist>(1),
    loop);

  if (np != 1)
    test_distributed_view<TestImplTag, T>(
      Domain<1>(16),
      Map<Block_dist>(1),
      Map<Block_dist>(np),
      loop);

#if VSIP_DIST_LEVEL >= 3
  test_distributed_view<TestImplTag, T>(
    Domain<1>(16),
    Map<Block_dist>(Block_dist(np)),
    Map<Cyclic_dist>(Cyclic_dist(np)),
    loop);

  test_distributed_view<TestImplTag, T>(
    Domain<1>(16),
    Map<Cyclic_dist>(Cyclic_dist(np)),
    Map<Block_dist>(Block_dist(np)),
    loop);

  test_distributed_view<TestImplTag, T>(
    Domain<1>(256),
    Map<Cyclic_dist>(Cyclic_dist(np, 4)),
    Map<Cyclic_dist>(Cyclic_dist(np, 3)),
    loop);
#endif
}



// Test several distributed matrix cases for a given type and parallel
// assignment implementation.

template <typename TestImplTag,
	  typename T>
void
test_matrix(int loop)
{
  length_type np, nr, nc;
  get_np_half(np, nr, nc);

  test_distributed_view<TestImplTag, T>(
    Domain<2>(4, 4),
    Map<>(Block_dist(np), Block_dist(1)),
    Map<>(Block_dist(1),  Block_dist(np)),
    loop);

  test_distributed_view<TestImplTag, T>(
    Domain<2>(16, 16),
    Map<>(Block_dist(nr), Block_dist(nc)),
    Map<>(Block_dist(nc), Block_dist(nr)),
    loop);
			  
#if VSIP_DIST_LEVEL >= 3
  test_distributed_view<TestImplTag, T>(
    Domain<2>(16, 16),
    Map<Cyclic_dist, Block_dist>(Cyclic_dist(nr, 2), Block_dist(nc)),
    Map<Block_dist, Cyclic_dist>(Block_dist(nc),    Cyclic_dist(nr, 2)),
    loop);
#endif

#if 0
  test_distributed_view<TestImplTag, T>(
    Domain<2>(256, 256),
    Map<>(Block_dist(nr), Block_dist(nc)),
    Map<>(Block_dist(nc), Block_dist(nr)),
    loop);
#endif
}



int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  int loop = argc > 1 ? atoi(argv[1]) : 1;

#if 0
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
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

  length_type np, nc, nr;
  get_np_square(np, nc, nr);

  test_vector<TestImplicit, float>(loop);
  test_matrix<TestImplicit, float>(loop);


  test_vector<TestExplicit<IMPL_TAG>, float>(loop);
  test_matrix<TestExplicit<IMPL_TAG>, float>(loop);

  test_local_view<float>(Domain<1>(2), Map<>(Block_dist(np)));
  test_local_view<float>(Domain<1>(2), Map<>(Block_dist(np > 1 ? np-1 : 1)));



  test_vector<TestImplicit, complex<float> >(loop);
  test_matrix<TestImplicit, complex<float> >(loop);

  test_vector<TestExplicit<IMPL_TAG>, complex<float> >(loop);
  test_matrix<TestExplicit<IMPL_TAG>, complex<float> >(loop);

  test_local_view<complex<float> >(Domain<1>(2), Map<>(Block_dist(np)));
  test_local_view<complex<float> >(Domain<1>(2), Map<>(np > 1 ? np-1 : 1));

#if TEST_OLD_PAR_ASSIGN
  // Enable this to test older assignments.
  test_vector<TestExplicit<impl::Packed_parallel_assign>,   float>(loop);
  test_matrix<TestExplicit<impl::Packed_parallel_assign>,   float>(loop);

  test_vector<TestExplicit<impl::Simple_parallel_assign_SOL>, float>(loop);
  test_vector<TestExplicit<impl::Simple_parallel_assign_DOL>, float>(loop);

  test_matrix<TestExplicit<impl::Simple_parallel_assign_SOL>, float>(loop);
  test_matrix<TestExplicit<impl::Simple_parallel_assign_DOL>, float>(loop);

  test_vector<TestExplicit<impl::Simple_parallel_assign_SOL>, int>(loop);
  test_vector<TestExplicit<impl::Simple_parallel_assign_DOL>, int>(loop);

  test_matrix<TestExplicit<impl::Simple_parallel_assign_SOL>, int>(loop);
  test_matrix<TestExplicit<impl::Simple_parallel_assign_DOL>, int>(loop);
#endif

  return 0;
}
