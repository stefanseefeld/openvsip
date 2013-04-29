//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Unit tests for parallel assignment.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/tensor.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "util.hpp"
#include "util-par.hpp"

using namespace vsip;

using vsip::impl::Length;
using vsip::impl::extent;

// Test a single parallel assignment.

template <typename                  T,
	  dimension_type            Dim,
	  typename                  Map1,
	  typename                  Map2>
void
test_par_assign(
  Domain<Dim> dom,
  Map1        map1,
  Map2        map2,
  int         loop)

{
  typedef typename impl::Row_major<Dim>::type order_type;

  typedef Dense<Dim, T, order_type, Map1>   dist_block1_t;
  typedef Dense<Dim, T, order_type, Map2>   dist_block2_t;

  typedef typename impl::view_of<dist_block1_t>::type view1_t;
  typedef typename impl::view_of<dist_block2_t>::type view2_t;

  view1_t view1(create_view<view1_t>(dom, T(), map1));
  view2_t view2(create_view<view2_t>(dom, T(), map2));

  check_local_view<Dim>(view1);
  check_local_view<Dim>(view2);

  foreach_point(view1, Set_identity<Dim>(dom));
  for (int l=0; l<loop; ++l)
  {
    view2 = view1;
  }
  foreach_point(view2, Check_identity<Dim>(dom));
}



template <typename T>
void
test_par_assign_cases(int loop)
{
  length_type np, nr, nc;
  get_np_square(np, nr, nc);

  // Vector Serial -> Serial
  // std::cout << "Replicated_map<1> -> Global_map<1>\n" << std::flush;
  test_par_assign<float>(Domain<1>(16),
			 Replicated_map<1>(),
			 Replicated_map<1>(),
			 loop);

  // Vector Serial -> Block_dist
  // std::cout << "Replicated_map<1> -> Map<Block_dist>\n" << std::flush;
  test_par_assign<float>(Domain<1>(16),
			 Replicated_map<1>(),
			 Map<Block_dist>(Block_dist(np)),
			 loop);

  // Vector Block_dist -> Serial
  // std::cout << "Map<Block_dist> -> Replicated_map<1>\n" << std::flush;
  test_par_assign<float>(Domain<1>(16),
			 Map<Block_dist>(Block_dist(np)),
			 Replicated_map<1>(),
			 loop);

  // Matrix Serial -> Serial
  // std::cout << "Replicated_map<2> -> Replicated_map<2>\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Replicated_map<2>(),
			 Replicated_map<2>(),
			 loop);

  // Matrix Serial -> Block_dist
  // std::cout << "Replicated_map<2> -> Map<> (square)\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Replicated_map<2>(),
			 Map<Block_dist>(Block_dist(nr), Block_dist(nc)),
			 loop);
  // std::cout << "Replicated_map<2> -> Map<> (cols)\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Replicated_map<2>(),
			 Map<Block_dist>(Block_dist(1), Block_dist(np)),
			 loop);
  // std::cout << "Replicated_map<2> -> Map<> (rows)\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Replicated_map<2>(),
			 Map<Block_dist>(Block_dist(np), Block_dist(1)),
			 loop);

  // Matrix Block_dist -> Serial
  // std::cout << "Map<> (square) -> Replicated_map<2>\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Map<Block_dist>(Block_dist(nr), Block_dist(nc)),
			 Replicated_map<2>(),
			 loop);
  // std::cout << "Map<> (cols) -> Replicated_map<2>\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Map<Block_dist>(Block_dist(1), Block_dist(np)),
			 Replicated_map<2>(),
			 loop);
  // std::cout << "Map<> (rows) -> Replicated_map<2>\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Map<Block_dist>(Block_dist(np), Block_dist(1)),
			 Replicated_map<2>(),
			 loop);

  // std::cout << "Map<> (rows) -> Map<> (cols)\n" << std::flush;
  test_par_assign<float>(Domain<2>(16, 16),
			 Map<Block_dist>(Block_dist(np), Block_dist(1)),
			 Map<Block_dist>(Block_dist(1), Block_dist(np)),
			 loop);

  // Tensor case.
  // std::cout << "3D: Map<> (rows) -> Map<> (cols)\n" << std::flush;
  test_par_assign<float>(Domain<3>(16, 8, 5),
		Map<Block_dist>(Block_dist(np), Block_dist(1), Block_dist(1)),
		Map<Block_dist>(Block_dist(1), Block_dist(np), Block_dist(1)),
		loop);
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

  test_par_assign_cases<float>(loop);
  test_par_assign_cases<complex<float> >(loop);

  return 0;
}
