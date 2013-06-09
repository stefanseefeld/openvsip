//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_map_utils_hpp_
#define ovxx_parallel_map_utils_hpp_

#include <vsip/map.hpp>
#include <vsip/impl/dist.hpp>

namespace ovxx
{
namespace parallel
{
template <typename D0, typename D1, typename D2>
struct map_project_1<0, Map<D0, D1, D2> >
{
  typedef Map<D1, D2> type;

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx,
				    index_type sb)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx);

    return fix_sb_0*num_sb_1*num_sb_2+ sb;
  }

  static type project(Map<D0, D1, D2> const &map, index_type idx)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx);

    vsip::Vector<processor_type> pvec(num_sb_1*num_sb_2);

    for (index_type pi=0; pi<num_sb_1*num_sb_2; ++pi)
      pvec.put(pi, map.impl_proc_from_rank(fix_sb_0*num_sb_1*num_sb_2+pi));

    return type(pvec, copy_dist<D1>(map, 1), copy_dist<D2>(map, 2));
  }
};

template <typename D0, typename D1, typename D2>
struct map_project_1<1, Map<D0, D1, D2> >
{
  typedef Map<D0, D2> type;

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx,
				    index_type sb)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx);

    index_type sb_0 = sb / num_sb_2;
    index_type sb_2 = sb % num_sb_2;

    return sb_0*num_sb_1*num_sb_2 + fix_sb_1*num_sb_2      + sb_2;
  }

  static type project(Map<D0, D1, D2> const& map, index_type idx)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx);

    vsip::Vector<processor_type> pvec(num_sb_0*num_sb_2);

    for (index_type pi=0; pi<num_sb_0*num_sb_2; ++pi)
    {
      index_type sb_0 = pi / num_sb_2;
      index_type sb_2 = pi % num_sb_2;
      pvec(pi) = map.impl_proc_from_rank(sb_0*num_sb_1*num_sb_2 +
					 fix_sb_1*num_sb_2      + sb_2);
    }

    return type(pvec, copy_dist<D1>(map, 0), copy_dist<D2>(map, 2));
  }
};

template <typename D0, typename D1, typename D2>
struct map_project_1<2, Map<D0, D1, D2> >
{
  typedef Map<D0, D1> type;

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx,
				    index_type sb)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx);

    index_type sb_0 = sb / num_sb_1;
    index_type sb_1 = sb % num_sb_1;
    return sb_0*num_sb_1*num_sb_2 + sb_1*num_sb_2          + fix_sb_2;
  }

  static type project(Map<D0, D1, D2> const& map, index_type idx)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx);

    vsip::Vector<processor_type> pvec(num_sb_0*num_sb_1);

    for (index_type pi=0; pi<num_sb_0*num_sb_1; ++pi)
    {
      index_type sb_0 = pi / num_sb_1;
      index_type sb_1 = pi % num_sb_1;
      pvec(pi) = map.impl_proc_from_rank(sb_0*num_sb_1*num_sb_2 +
					 sb_1*num_sb_2          +
					 fix_sb_2);
    }

    return type(pvec, copy_dist<D1>(map, 0), copy_dist<D2>(map, 1));
  }
};

template <typename D0, typename D1, typename D2>
struct map_project_2<0, 1, Map<D0, D1, D2> >
{
  typedef Map<D2> type;

  static type project(Map<D0, D1, D2> const& map,
		      index_type idx0,
		      index_type idx1)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);

    vsip::Vector<processor_type> pvec(num_sb_2);

    for (index_type pi=0; pi<num_sb_2; ++pi)
      pvec(pi) = map.impl_proc_from_rank(fix_sb_0*num_sb_1*num_sb_2 +
					 fix_sb_1*num_sb_2 + pi);

    return type(pvec, copy_dist<D2>(map, 2));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx0,
				    index_type idx1,
				    index_type sb)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);

    return fix_sb_0*num_sb_1*num_sb_2 + fix_sb_1*num_sb_2 + sb;
  }
};

template <typename D0, typename D1, typename D2>
struct map_project_2<0, 2, Map<D0, D1, D2> >
{
  typedef Map<D2> type;

  static type project(Map<D0, D1, D2> const& map,
		      index_type idx0,
		      index_type idx2)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    vsip::Vector<processor_type> pvec(num_sb_1);

    for (index_type pi=0; pi<num_sb_1; ++pi)
      pvec(pi) = map.impl_proc_from_rank(fix_sb_0*num_sb_1*num_sb_2 +
					 pi*num_sb_2 + fix_sb_2);

    return type(pvec, copy_dist<D1>(map, 1));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx0,
				    index_type idx2,
				    index_type sb)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    return fix_sb_0*num_sb_1*num_sb_2 + sb*num_sb_2 + fix_sb_2;
  }
};

template <typename D0, typename D1, typename D2>
struct map_project_2<1, 2, Map<D0, D1, D2> >
{
  typedef Map<D2> type;

  static type project(Map<D0, D1, D2> const& map,
		      index_type idx1,
		      index_type idx2)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    vsip::Vector<processor_type> pvec(num_sb_0);

    for (index_type pi=0; pi<num_sb_0; ++pi)
      pvec(pi) = map.impl_proc_from_rank(pi*num_sb_1*num_sb_2 +
					 fix_sb_1*num_sb_2 + fix_sb_2);

    return type(pvec, copy_dist<D0>(map, 0));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx1,
				    index_type idx2,
				    index_type sb)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    return sb*num_sb_1*num_sb_2 + fix_sb_1*num_sb_2 + fix_sb_2;
  }
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
