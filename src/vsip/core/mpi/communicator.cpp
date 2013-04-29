/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#include <vsip/core/mpi/communicator.hpp>

namespace vsip
{
namespace impl
{
namespace mpi
{

struct comm_free
{
  void operator()(MPI_Comm* comm) const
  {
    int finalized;
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Finalized, (&finalized));
    if (!finalized)
      VSIP_IMPL_MPI_CHECK_RESULT(MPI_Comm_free, (comm));
    delete comm;
  }
};

Communicator::Communicator()
{
  impl_.reset(new MPI_Comm(MPI_COMM_WORLD));
}

Communicator::Communicator(MPI_Comm const & comm, comm_create_kind kind)
{
  if (comm == MPI_COMM_NULL)
    /* MPI_COMM_NULL indicates that the communicator is not usable. */
    return;
  switch (kind) 
  {
    case comm_duplicate:
    {
      MPI_Comm newcomm;
      VSIP_IMPL_MPI_CHECK_RESULT(MPI_Comm_dup, (comm, &newcomm));
      impl_.reset(new MPI_Comm(newcomm), comm_free());
      MPI_Errhandler_set(newcomm, MPI_ERRORS_RETURN);
      break;
    }

    case comm_take_ownership:
      impl_.reset(new MPI_Comm(comm), comm_free());
      break;

    case comm_attach:
      impl_.reset(new MPI_Comm(comm));
      break;
  }
  length_type s = size();
  pvec_.resize(s);
  for (index_type i = 0; i < s; ++i)
    pvec_[i] = static_cast<processor_type>(i);
}

Communicator::Communicator(Communicator const &comm, Group const &subgroup)
{
  MPI_Comm newcomm;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Comm_create, (comm, subgroup, &newcomm));
  if (newcomm != MPI_COMM_NULL)
    impl_.reset(new MPI_Comm(newcomm), comm_free());

  if (*this) // only if this is a valid communicator
  {
    length_type s = size();
    pvec_.resize(s);
    for (index_type i = 0; i < s; ++i)
      pvec_[i] = static_cast<processor_type>(i);
  }
}

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip
