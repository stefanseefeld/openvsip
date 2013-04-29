//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <vsip/core/mpi/group.hpp>

namespace vsip
{
namespace impl
{
namespace mpi
{

struct group_free
{
  void operator()(MPI_Group* comm) const
  {
    int finalized;
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Finalized, (&finalized));
    if (!finalized)
      VSIP_IMPL_MPI_CHECK_RESULT(MPI_Group_free, (comm));
    delete comm;
  }
};


Group::Group(MPI_Group const &g, bool adopt)
{
  if (g != MPI_GROUP_EMPTY) 
  {
    if (adopt) impl_.reset(new MPI_Group(g), group_free());
    else       impl_.reset(new MPI_Group(g));
  }
}


} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip
