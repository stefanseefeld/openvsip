//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>
#include <vsip_csl/assignment.hpp>
#include <vsip_csl/test.hpp>
#include "util.hpp"

using namespace vsip;
using vsip_csl::equal;

/***********************************************************************
  Definitions
***********************************************************************/

// test1: 
//  - Create distribute user-storage block with NULL ptr,
//  - Bind buffers (using get_subblock_dom to determine size).
//  -

template <typename       T,
	  dimension_type Dim,
	  typename       MapT>
void
test1(
  Domain<Dim> const& dom,
  MapT               dist_map,
  bool               use_sa)
{
  typedef Map<Block_dist, Block_dist> root_map_t;

  typedef typename impl::Row_major<Dim>::type order_type;

  typedef Dense<Dim, T, order_type, root_map_t> root_block_t;
  typedef Dense<Dim, T, order_type, MapT>       dist_block_t;

  typedef typename impl::view_of<root_block_t>::type root_view_t;
  typedef typename impl::view_of<dist_block_t>::type dist_view_t;


  root_map_t  root_map(Block_dist(1), Block_dist(1));

  // root: root view.
  root_view_t root(create_view<root_view_t>(dom, T(0), root_map));

  // dist: distributed view, w/user-storage.
  dist_view_t dist(create_view<dist_view_t>(dom, static_cast<T*>(0),dist_map));

  vsip_csl::Assignment root_dist(root, dist);
  vsip_csl::Assignment dist_root(dist, root);

  // Initially, dist is not admitted.
  test_assert(dist.block().admitted() == false);

  // Find out how big the local subdomain is.
  Domain<Dim> subdom = subblock_domain(dist);

  // cout << "size: " << subdom.size() << endl;

  length_type loop = 5;
  T** data = new T*[loop];

  for (index_type iter=0; iter<loop; ++iter)
  {
    // cout << "iter " << iter << endl;

    // allocate data that this processor owns.
    data[iter] = 0;
    if (subdom.size() > 0)
      data[iter] = new T[subdom.size()];

    dist.block().rebind(data[iter]);

    // Put some data in buffer.
    for (index_type p=0; p<num_patches(dist); ++p)
    {
      Domain<Dim> l_dom = local_domain(dist, p);
      Domain<Dim> g_dom = global_domain(dist, p);

      for (index_type i=0; i<l_dom[0].size(); ++i)
      {
	index_type li = l_dom.impl_nth(i);
	index_type gi = g_dom.impl_nth(i);

	// cout << "  data[" << li << "] = " << T(iter*gi) << endl;

	data[iter][li] = T(iter*gi);
      }
    }

    // admit the block
    dist.block().admit(true);
    test_assert(dist.block().admitted() == true);

    // assign to root
    if (use_sa)
      root_dist();
    else
      root = dist;

    // On the root processor ...
    if (root_map.subblock() != no_subblock)
    {
      typename root_view_t::local_type l_root = root.local();

      // ... check that root is correct.
      for (index_type i=0; i<l_root.size(); ++i)
	test_assert(equal(l_root(i), T(iter*i)));

      // ... set values for the round trip
      for (index_type i=0; i<l_root.size(); ++i)
	l_root(i) = T(iter*i+1);
    }

    // assign back to dist
    if (use_sa)
      dist_root();
    else
      dist = root;

    // release the block
    dist.block().release(true);
    test_assert(dist.block().admitted() == false);

    // Check the data in buffer.
    for (index_type p=0; p<num_patches(dist); ++p)
    {
      Domain<Dim> l_dom = local_domain(dist, p);
      Domain<Dim> g_dom = global_domain(dist, p);
      
      for (index_type i=0; i<l_dom[0].size(); ++i)
      {
	index_type li = l_dom.impl_nth(i);
	index_type gi = g_dom.impl_nth(i);
	
	test_assert(equal(data[iter][li], T(iter*gi+1)));
      }
    }
  }

  for (index_type iter=0; iter<loop; ++iter)
    if (data[iter]) delete[] data[iter];
  delete[] data;
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if 0
  // Enable this section for easier debugging.
  impl::Communicator comm = impl::default_communicator();
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

  processor_type np = num_processors();

  Map<Block_dist>  map1 = Map<Block_dist>(Block_dist(np));
  test1<float>(Domain<1>(10), map1, false);

#if VSIP_DIST_LEVEL >= 3
  Map<Cyclic_dist> map2 = Map<Cyclic_dist>(Cyclic_dist(np));
  test1<float>(Domain<1>(10), map2, false);
#endif
}
