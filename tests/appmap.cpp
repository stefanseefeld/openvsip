//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/matrix.hpp>
#include <vsip/initfin.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;

using vsip::impl::Length;
using vsip::impl::extent;
using vsip::impl::valid;
using vsip::impl::next;
using vsip::impl::domain_nth;



/***********************************************************************
  Definitions
***********************************************************************/


/// Get the nth index in a domain.


// Utility to create a processor vector of given size.

// Requires:
//   NUM_PROC is number of processors to place in vector.
//
// Returns:
//   Vector with NUM_PROC processors.

Vector<processor_type>
create_pvec(
  length_type num_proc)
{
  Vector<processor_type> pvec(num_proc);

  for (index_type i=0; i<num_proc; ++i)
    pvec.put(i, (i+1)*10);

  return pvec;
}
  


// Dump appmap to stream.

template <typename MapT>
void
dump_appmap(
  std::ostream&          out,
  MapT const&            map,
  Vector<processor_type> pvec)
{
  out << "App_map:\n"
      << "   Dim: (" << map.num_subblocks(0) << " x "
                     << map.num_subblocks(1) << " x "
                     << map.num_subblocks(2) << ")\n"
      << "-----------"
      << endl;

  for (index_type pi=0; pi<pvec.size(); ++pi)
  {
    processor_type pr = pvec.get(pi);
    index_type     sb = map.subblock(pr);
    
    if (sb != no_subblock)
    {
      for (index_type p=0; p<map.impl_num_patches(sb); ++p)
      {
	Domain<3> gdom = map.template impl_global_domain<3>(sb, p);
	Domain<3> ldom = map.template impl_local_domain<3>(sb, p);
	out << "  pr=" << pr << "  sb=" << sb << " patch=" << p
	    << "  gdom=" << gdom
	    << "  ldom=" << ldom
	    << endl;
      }
    }
  }
}



// Check that local and global indices within a patch are consistent.

template <dimension_type Dim,
	  typename       MapT>
void
check_local_vs_global(
  MapT const&   map,
  index_type    sb,
  index_type    p)
{
  Domain<Dim> gdom = map.template impl_global_domain<Dim>(sb, p);
  Domain<Dim> ldom = map.template impl_local_domain<Dim>(sb, p);

  test_assert(gdom.size() == ldom.size());

  length_type dim_num_subblocks[Dim]; // number of subblocks in each dim
  length_type dim_sb[Dim];            // local sb in each dim
  length_type dim_num_patches[Dim];   // number of patches in each dim
  length_type dim_p[Dim];             // local p in each dim

  for (dimension_type d=0; d<Dim; ++d)
    dim_num_subblocks[d] = map.num_subblocks(d);
  
  vsip::impl::split_tuple(sb, Dim, dim_num_subblocks, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
    dim_num_patches[d] = map.impl_subblock_patches(d, dim_sb[d]);

  vsip::impl::split_tuple(p, Dim, dim_num_patches, dim_p);

  for (dimension_type d=0; d<Dim; ++d)
  {
    test_assert(gdom[d].length() == ldom[d].length());

    for (index_type i=0; i<ldom[d].length(); ++i)
    {
      index_type gi = gdom[d].impl_nth(i);
      index_type li = ldom[d].impl_nth(i);

      test_assert(map.impl_local_from_global_index(d, gi) == li);
      test_assert(map.impl_global_from_local_index(d, sb, li) == gi);
      test_assert(map.impl_dim_subblock_from_index(d, gi) == dim_sb[d]);
      test_assert(map.impl_dim_patch_from_index(d, gi) == dim_p[d]);
    }
  }


  Length<Dim> ext = extent(gdom);
  for(Index<Dim> idx; valid(ext,idx); next(ext,idx))
  {
    Index<Dim> g_idx = domain_nth(gdom,idx);
    Index<Dim> l_idx = domain_nth(ldom,idx);
    test_assert(map.impl_subblock_from_global_index(g_idx) == sb);
    test_assert(map.impl_patch_from_global_index(g_idx)    == p);
  }

}



// Test 1-dimensional applied map.

// Checks that each index in an applied map's distributed domain is in
// one and only one subblock-patch.

template <typename Dist0>
void
tc_appmap(
  length_type num_proc,
  Domain<1>   dom,
  Dist0       dist0)
{
  dimension_type const dim = 1;

  typedef Map<Dist0> map_t;

  Vector<processor_type> pvec = create_pvec(num_proc);

  map_t map(pvec, dist0);
  map.impl_apply(Domain<dim>(dom.length()));

  Vector<int> data(dom.length(), 0);

  for (index_type pi=0; pi<pvec.size(); ++pi)
  {
    processor_type pr = pvec.get(pi);
    index_type     sb = map.subblock(pr);
    
    if (sb != no_subblock)
    {
      for (index_type p=0; p<map.impl_num_patches(sb); ++p)
      {
	Domain<dim> gdom = map.template impl_global_domain<dim>(sb, p);

	if (gdom.size() > 0)
	  data(gdom) += 1;

	check_local_vs_global<dim>(map, sb, p);
      }
    }
  }

  // Check that every element in vector was marked, once.
  for (index_type i=0; i<data.size(); ++i)
    test_assert(data.get(i) == 1);
}



// Test 2-dimensional applied Map.

// Checks that each index in an applied Map's distributed domain is in one
// and only one subblock-patch.

template <typename Dist0,
	  typename Dist1>
void
tc_appmap(
  length_type num_proc,
  Domain<2>   dom,
  Dist0       dist0,
  Dist1       dist1)
{
  dimension_type const dim = 2;

  typedef Map<Dist0, Dist1> map_t;

  Vector<processor_type> pvec = create_pvec(num_proc);

  map_t map(pvec, dist0, dist1);
  map.impl_apply(dom);

  Matrix<int> data(dom[0].length(), dom[1].length(), 0);

  for (index_type pi=0; pi<pvec.size(); ++pi)
  {
    processor_type pr = pvec.get(pi);
    index_type     sb = map.subblock(pr);
    
    if (sb != no_subblock)
    {
      for (index_type p=0; p<map.impl_num_patches(sb); ++p)
      {
	Domain<dim> gdom = map.template impl_global_domain<dim>(sb, p);

	data(gdom) += 1;

	check_local_vs_global<dim>(map, sb, p);
      }
    }
  }

  // cout << "tc_appmap:\n" << data;

  // Check that every element in vector was marked, once.
  for (index_type r=0; r<data.size(0); ++r)
    for (index_type c=0; c<data.size(1); ++c)
      test_assert(data.get(r, c) == 1);
}



// Test various 1-dim cases.

void
test_1d_appmap()
{
  tc_appmap(4, 3, Block_dist(4));

  tc_appmap(4, 16, Block_dist(4));
  tc_appmap(4, 17, Block_dist(4));
  tc_appmap(4, 18, Block_dist(4));
  tc_appmap(4, 19, Block_dist(4));

  tc_appmap(4, 15, Cyclic_dist(4, 1));
  tc_appmap(4, 15, Cyclic_dist(4, 2));
  tc_appmap(4, 15, Cyclic_dist(4, 3));
  tc_appmap(4, 15, Cyclic_dist(4, 4));

  tc_appmap(4, 16, Cyclic_dist(4, 1));
  tc_appmap(4, 16, Cyclic_dist(4, 2));
  tc_appmap(4, 16, Cyclic_dist(4, 3));
  tc_appmap(4, 16, Cyclic_dist(4, 4));

  tc_appmap(4, 17, Cyclic_dist(4, 1));
  tc_appmap(4, 17, Cyclic_dist(4, 2));
  tc_appmap(4, 17, Cyclic_dist(4, 3));
  tc_appmap(4, 17, Cyclic_dist(4, 4));

  tc_appmap(4, 16, Cyclic_dist(3, 1));
  tc_appmap(4, 16, Cyclic_dist(3, 2));
  tc_appmap(4, 16, Cyclic_dist(3, 3));
  tc_appmap(4, 16, Cyclic_dist(3, 4));
}



// Test various 2-dim cases.

void
test_2d_appmap()
{
  tc_appmap(16, Domain<2>(16, 16), Block_dist(4), Block_dist(4));
  tc_appmap(16, Domain<2>(17, 18), Block_dist(4), Block_dist(4));
  tc_appmap(16, Domain<2>(16, 19), Block_dist(4), Block_dist(4));

  for (index_type i=16; i<20; ++i)
    tc_appmap(4, Domain<2>(i, 16), Block_dist(4), Block_dist(1));

  for (index_type i=16; i<20; ++i)
  {
    tc_appmap(12, Domain<2>(i, i+1), Cyclic_dist(4, 1), Cyclic_dist(3, 4));
    tc_appmap(12, Domain<2>(i, i+1), Cyclic_dist(4, 2), Cyclic_dist(3, 3));
    tc_appmap(12, Domain<2>(i, i+1), Cyclic_dist(4, 3), Cyclic_dist(3, 2));
    tc_appmap(12, Domain<2>(i, i+1), Cyclic_dist(4, 4), Cyclic_dist(3, 1));
  }
}




void
test_appmap()
{
  typedef Map<Block_dist, Block_dist> map_t;

  length_type const num_proc = 16;

  Vector<processor_type> pvec = create_pvec(num_proc);

  map_t map(pvec, Block_dist(4), Block_dist(4));
  map.impl_apply(Domain<3>(16, 16, 1));

  test_assert(map.impl_num_patches(0) == 1);
  test_assert(map.impl_global_domain<3>(0, 0) ==
	      Domain<3>(Domain<1>(0, 1, 4),
			Domain<1>(0, 1, 4),
			Domain<1>(0, 1, 1)));

  // subblocks are row-major
  test_assert(map.impl_num_patches(1) == 1);
  test_assert(map.impl_global_domain<3>(1, 0) ==
	      Domain<3>(Domain<1>(0, 1, 4),
			Domain<1>(4, 1, 4),
			Domain<1>(0, 1, 1)));

  test_assert(map.impl_num_patches(15) == 1);
  test_assert(map.impl_global_domain<3>(15, 0) ==
	      Domain<3>(Domain<1>(12, 1, 4),
			Domain<1>(12, 1, 4),
			Domain<1>(0, 1, 1)));
}



// Test what happens when number of subblocks > elements, forcing
// multiple subblocks to be empty.
void
test_empty_subblocks()
{
  typedef Map<Block_dist> map_t;

  length_type const subblocks  = 6;
  length_type const size       = 4;

  Vector<processor_type> pvec = create_pvec(subblocks);

  map_t map(pvec, Block_dist(subblocks));
  map.impl_apply(Domain<1>(4));

  length_type sum = 0;
  for (index_type i=0; i<subblocks; ++i)
  {
    std::cout << " i = " << map.impl_subblock_domain<1>(i).size() << std::endl;
    test_assert(map.impl_subblock_domain<1>(i).size() == 1 ||
		map.impl_subblock_domain<1>(i).size() == 0);
    sum += map.impl_subblock_domain<1>(i).size();
  }
  std::cout << "sum = " << sum << std::endl;
  test_assert(sum == size);
}



int
main()
{
  vsip::vsipl init;

  test_appmap();

  test_empty_subblocks();

  test_1d_appmap();
  test_2d_appmap();
}
