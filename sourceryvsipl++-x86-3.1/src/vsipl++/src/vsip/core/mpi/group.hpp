/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#ifndef vsip_core_mpi_group_hpp_
#define vsip_core_mpi_group_hpp_

#include <vsip/support.hpp>
#include <vsip/core/mpi/exception.hpp>
#include <vsip/core/c++0x.hpp>
#include <vector>
#include <mpi.h>

namespace vsip
{
namespace impl
{
namespace mpi
{

/// A group is a representation of a subset of the processes
/// within a `Communicator`.
class Group
{
public:
  /// Construct an empty group.
  Group() {}
  Group(MPI_Group const &g, bool adopt = true);

  operator MPI_Group() const { return impl_.get() ? *impl_ : MPI_GROUP_EMPTY;}

  /// Determine the rank of the calling process in the group.
  index_type rank() const
  {
    if (!impl_.get()) return no_rank;
    int r;
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_rank, (*this, &r));
    return r == MPI_UNDEFINED ? no_rank : r;
  }
  /// Determine the number of processes in the group.
  length_type size() const
  {
    if (!impl_.get()) return 0;
    int s;
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_size, (*this, &s));
    return s;
  }

  /// Creates a new group including a subset of the processes
  /// in the current group.
  template<typename InputIterator>
  Group include(InputIterator first, InputIterator last);
  /// Creates a new group from all of the processes in the
  /// current group, exluding a specific subset of the processes.
  template<typename InputIterator>
  Group exclude(InputIterator first, InputIterator last);
  /// Translates the ranks from one group into the ranks of the
  /// same processes in another group.
  template<typename InputIterator, typename OutputIterator>
  OutputIterator translate_ranks(InputIterator first, InputIterator last,
                                 Group const &, OutputIterator out);

private:
  shared_ptr<MPI_Group> impl_;
};

template<typename InputIterator>
Group Group::include(InputIterator first, InputIterator last)
{
  if (first == last) return Group();
  std::vector<int> ranks(first, last);
  MPI_Group g;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_incl, (*this, ranks.size(), &ranks[0], &g));
  return Group(g);
}

template<> 
inline Group 
Group::include(int *first, int *last)
{
  MPI_Group g;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_incl, (*this, last - first, first, &g));
  return Group(g);
}

template<typename InputIterator>
Group Group::exclude(InputIterator first, InputIterator last)
{
  if (first == last) return Group();
  std::vector<int> ranks(first, last);
  MPI_Group g;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_excl, (*this, ranks.size(), &ranks[0], &g));
  return Group(g);
}

template<> 
inline Group 
Group::exclude(int *first, int *last)
{
  MPI_Group g;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_excl, (*this, last - first, first, &g));
  return Group(g);
}

template<typename InputIterator, typename OutputIterator>
OutputIterator 
Group::translate_ranks(InputIterator first, InputIterator last,
                       Group const &to_group, OutputIterator out)
{
  std::vector<int> in_array(first, last);
  if (in_array.empty())
    return out;

  std::vector<int> out_array(in_array.size());
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_translate_ranks,
			     (*this, in_array.size(), &in_array[0],
			      to_group, &out_array[0]));
  for (std::vector<int>::size_type i = 0, n = out_array.size(); i < n; ++i)
    *out++ = out_array[i];
  return out;
}

template<> 
inline int*
Group::translate_ranks(int *first, int *last, Group const &to_group, int *out)
{
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_translate_ranks,
			     (*this, last-first, first, to_group, out));
  return out + (last - first);
}

inline Group operator|(Group const &g1, Group const &g2)
{
  MPI_Group result;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_union, (g1, g2, &result));
  return Group(result);
}

inline Group operator&(Group const &g1, Group const &g2)
{
  MPI_Group result;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_intersection, (g1, g2, &result));
  return Group(result);
}

inline Group operator-(Group const &g1, Group const &g2)
{
  MPI_Group result;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_difference, (g1, g2, &result));
  return Group(result);
}

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip

#endif
