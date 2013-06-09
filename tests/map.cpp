//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cassert>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/initfin.hpp>
#include <test.hpp>

using namespace ovxx;
namespace p = ovxx::parallel;

// Check a distribution against expected type, num_subblocks, and
// contiguity.

template <typename Dist>
void
check_distribution(Dist const& dist,
		   distribution_type type,
		   length_type num_subblocks,
		   length_type contiguity)
{
  test_assert(dist.distribution()      == type);
  test_assert(dist.num_subblocks()     == num_subblocks);
  test_assert(dist.cyclic_contiguity() == contiguity);
}



// Test Block_dist distribution.

void
test_block_distribution()
{
  // Test default constructor
  Block_dist dist1;
  check_distribution(dist1, block,  1, 0);

  // Test num_subblock constructor
  Block_dist dist2(3);
  check_distribution(dist2, block,  3, 0);

  // Test copy-constructor
  Block_dist dist3(dist2);
  check_distribution(dist3, block,  3, 0);

  // Test assignement
  dist1 = dist3;
  check_distribution(dist1, block,  3, 0);

  // Do a few more tests.
  check_distribution(Block_dist(),   block,  1, 0);
  check_distribution(Block_dist(3),  block,  3, 0);
  check_distribution(Block_dist(32), block, 32, 0);
}



// Test Cyclic_dist distribution.

void
test_cyclic_distribution()
{
  // Test default constructor
  Cyclic_dist dist1;
  check_distribution(dist1, cyclic,  1, 1);

  // Test num_subblock constructor
  Cyclic_dist dist2(3);
  check_distribution(dist2, cyclic,  3, 1);

  // Test num_subblock, contiguity constructor
  Cyclic_dist dist3(3, 2);
  check_distribution(dist3, cyclic,  3, 2);

  // Test copy-constructor
  Cyclic_dist dist4(dist3);
  check_distribution(dist3, cyclic,  3, 2);

  // Test assignement
  dist1 = dist2;
  check_distribution(dist1, cyclic,  3, 1);

  // Do a few more tests.
  check_distribution(Cyclic_dist(),   cyclic,  1, 1);
  check_distribution(Cyclic_dist(3),  cyclic,  3, 1);
  check_distribution(Cyclic_dist(32), cyclic, 32, 1);

  check_distribution(Cyclic_dist(5,   3), cyclic,  5,  3);
  check_distribution(Cyclic_dist(33, 11), cyclic, 33, 11);
}



// Basic tests for Map.

void
test_map_basic()
{
  // Test map with default dimension constructors.
  // Each dimension should default to 1 subblock.
  Map<Block_dist, Block_dist> map1;

  test_assert(map1.distribution(0)      == block);
  test_assert(map1.num_subblocks(0)     == 1);
  test_assert(map1.cyclic_contiguity(0) == 0);

  test_assert(map1.distribution(1)      == block);
  test_assert(map1.num_subblocks(1)     == 1);
  test_assert(map1.cyclic_contiguity(1) == 0);

  test_assert(map1.distribution(2)      == block);
  test_assert(map1.num_subblocks(2)     == 1);
  test_assert(map1.cyclic_contiguity(2) == 0);



  Map<Block_dist, Cyclic_dist> map2;

  test_assert(map2.distribution(0)      == block);
  test_assert(map2.num_subblocks(0)     == 1);
  test_assert(map2.cyclic_contiguity(0) == 0);

  test_assert(map2.distribution(1)      == cyclic);
  test_assert(map2.num_subblocks(1)     == 1);
  test_assert(map2.cyclic_contiguity(1) == 1);

  test_assert(map2.distribution(2)      == block);
  test_assert(map2.num_subblocks(2)     == 1);
  test_assert(map2.cyclic_contiguity(2) == 0);



  Vector<processor_type> pvec(12); pvec = 0;
  Map<Block_dist, Cyclic_dist> map3(pvec, Block_dist(3), Cyclic_dist(4, 2));

  test_assert(map3.distribution(0)      == block);
  test_assert(map3.num_subblocks(0)     == 3);
  test_assert(map3.cyclic_contiguity(0) == 0);

  test_assert(map3.distribution(1)      == cyclic);
  test_assert(map3.num_subblocks(1)     == 4);
  test_assert(map3.cyclic_contiguity(1) == 2);

  test_assert(map3.distribution(2)      == block);
  test_assert(map3.num_subblocks(2)     == 1);
  test_assert(map3.cyclic_contiguity(2) == 0);
}



// Count the number of subblocks in a range.

template <typename SubblockIterator>
length_type
count_subblocks(SubblockIterator begin,
		SubblockIterator end)
{
  int count = 0;
  for (SubblockIterator cur = begin; cur != end; ++cur)
    ++count;
  // SubblockIterator is Random Access Iterator
  test_assert(end - begin == count);
  return count;
}



// Check that a sequence of subblocks are all mapped to a processor pr.

template <typename Map>
void
check_subblock(Map const &map, processor_type pr, index_type sb)
{
  typedef typename Map::processor_iterator processor_iterator;

  if (sb != no_subblock)
  {
    processor_iterator pbegin = map.processor_begin(sb);
    processor_iterator pend   = map.processor_end(sb);

    length_type pr_count = 0;
    for (processor_iterator pcur = pbegin; pcur != pend; ++pcur)
    {
      ++pr_count;
      test_assert(*pcur == pr);
    }
    test_assert(pr_count == 1);
    test_assert(pend - pbegin == 1);

  }
}



// Utility to create a processor vector of given size.

// Requires:
//   NUM_PROC is number of processors to place in vector.
//
// Returns:
//   Vector with NUM_PROC processors.

Vector<processor_type>
create_pvec(length_type num_proc)
{
  Vector<processor_type> pvec(num_proc);

  for (index_type i=0; i<num_proc; ++i)
    pvec.put(i, (i+1)*10);

  return pvec;
}
  


// Testcase: test validity of map's subblocks.

// Requires:
//   NUM_PROC is the number of processors

template <typename Dist0,
	  typename Dist1>
void
tc_map_subblocks(length_type num_proc, Dist0 dist0, Dist1 dist1)
{
  typedef Map<Dist0, Dist1> map_t;

  Vector<processor_type> pvec = create_pvec(num_proc);

  length_type dim0 = dist0.num_subblocks();
  length_type dim1 = dist1.num_subblocks();
  length_type num_subblocks = dim0 * dim1;

  map_t map(pvec, dist0, dist1);

  length_type total = 0;	// total number of subblocks

  for (index_type i=0; i<num_proc; ++i)
  {
    processor_type pr = pvec.get(i);

    // Compute the number of subblocks expected on this processor:
    length_type expected_count = num_subblocks/num_proc +
				 (i < num_subblocks%num_proc ? 1 : 0);

    index_type sb = map.subblock(pr);

    length_type count = sb == no_subblock ? 0 : 1;

    // Check the number of subblocks per processor.
    test_assert(count == expected_count);

    // Check that each subblock is only mapped to this processr.
    check_subblock(map, pr, sb);

    total += count;
  }

  // Check that number of subblocks iterated over equals expected.
  test_assert(total == num_subblocks);
}



// Test various map distributions.

void
test_map_subblocks()
{
  tc_map_subblocks(16, Block_dist (4), Block_dist (4));
  tc_map_subblocks(16, Block_dist (4), Block_dist (4));
  tc_map_subblocks(20, Cyclic_dist(4), Block_dist (5));
  tc_map_subblocks(20, Block_dist (5), Cyclic_dist(4));
  tc_map_subblocks(9, Block_dist (3), Block_dist (3));
  tc_map_subblocks(16, Cyclic_dist(4), Cyclic_dist(4));
}



// Test segment size utility functions.

void
test_segment_size()
{
  // If size is a multiple of the number of segments, all segments
  // should be the same size.
  for (index_type i=0; i<5; ++i)
  {
    test_assert(p::segment_size(10, 5, i) == 2);
    test_assert(p::segment_size(10, 5, 1, i) == 2);
  }

  // Extra elements should be spread across the first segements.
  test_assert(p::segment_size(11, 5, 0) == 3);
  test_assert(p::segment_size(11, 5, 1) == 2);
  test_assert(p::segment_size(11, 5, 2) == 2);
  test_assert(p::segment_size(11, 5, 3) == 2);
  test_assert(p::segment_size(11, 5, 4) == 2);

  // Extra elements should be spread across the first segements.
  test_assert(p::segment_size(13, 5, 0) == 3);
  test_assert(p::segment_size(13, 5, 1) == 3);
  test_assert(p::segment_size(13, 5, 2) == 3);
  test_assert(p::segment_size(13, 5, 3) == 2);
  test_assert(p::segment_size(13, 5, 4) == 2);

  // Extra elements should be spread across the first segements.
  test_assert(p::segment_size(13, 5, 1, 0) == 3);
  test_assert(p::segment_size(13, 5, 1, 1) == 3);
  test_assert(p::segment_size(13, 5, 1, 2) == 3);
  test_assert(p::segment_size(13, 5, 1, 3) == 2);
  test_assert(p::segment_size(13, 5, 1, 4) == 2);

  // Check how chunksize of 2 is handled
  test_assert(p::segment_size(16, 5, 2, 0) == 4);
  test_assert(p::segment_size(16, 5, 2, 1) == 4);
  test_assert(p::segment_size(16, 5, 2, 2) == 4);
  test_assert(p::segment_size(16, 5, 2, 3) == 2);
  test_assert(p::segment_size(16, 5, 2, 4) == 2);

  test_assert(p::segment_size(14, 5, 2, 0) == 4);
  test_assert(p::segment_size(14, 5, 2, 1) == 4);
  test_assert(p::segment_size(14, 5, 2, 2) == 2);
  test_assert(p::segment_size(14, 5, 2, 3) == 2);
  test_assert(p::segment_size(14, 5, 2, 4) == 2);

  // Check how odd partial chunk is handled:
  test_assert(p::segment_size(15, 5, 2, 0) == 4);
  test_assert(p::segment_size(15, 5, 2, 1) == 4);
  test_assert(p::segment_size(15, 5, 2, 2) == 3);
  test_assert(p::segment_size(15, 5, 2, 3) == 2);
  test_assert(p::segment_size(15, 5, 2, 4) == 2);

  test_assert(p::segment_size(15, 4, 4, 0) == 4);
  test_assert(p::segment_size(15, 4, 4, 1) == 4);
  test_assert(p::segment_size(15, 4, 4, 2) == 4);
  test_assert(p::segment_size(15, 4, 4, 3) == 3);

  test_assert(p::segment_size(11, 4, 4, 0) == 4);
  test_assert(p::segment_size(11, 4, 4, 1) == 4);
  test_assert(p::segment_size(11, 4, 4, 2) == 3);
  test_assert(p::segment_size(11, 4, 4, 3) == 0);

  test_assert(p::segment_size(6, 4, 4, 0) == 4);
  test_assert(p::segment_size(6, 4, 4, 1) == 2);
  test_assert(p::segment_size(6, 4, 4, 2) == 0);
  test_assert(p::segment_size(6, 4, 4, 3) == 0);
}

int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  test_block_distribution();
  test_cyclic_distribution();

  test_map_basic();
  test_map_subblocks();

  test_segment_size();
}
