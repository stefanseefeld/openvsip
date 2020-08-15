//
// Copyright (c) 2010 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_mpi_communicator_hpp_
#define ovxx_mpi_communicator_hpp_

#include <ovxx/support.hpp>
#include <ovxx/reductions/types.hpp>
#include <ovxx/mpi/exception.hpp>
#include <ovxx/mpi/datatype.hpp>
#include <ovxx/mpi/group.hpp>
#include <ovxx/c++11.hpp>
#include <mpi.h>

namespace ovxx
{
namespace mpi
{

/// A communicator permits communication and
/// synchronization among a set of processes.
class Communicator
{
public:
  typedef MPI_Request  request_type;
  typedef MPI_Datatype chain_type;
  typedef std::vector<processor_type> pvec_type;

public:
  enum comm_create_kind { comm_duplicate, comm_take_ownership, comm_attach};

  Communicator();

  /// Build a new Communicator based on a subgroup of another
  /// Communicator.
  Communicator(Communicator const &comm, Group const &subgroup);

  Communicator(MPI_Comm const &c, comm_create_kind kind);

  operator MPI_Comm() const { return impl_.get() ? *impl_ : MPI_COMM_NULL;}
  operator bool() const { return impl_.get();}

  /// Determine the rank of the executing process in a
  /// communicator.
  processor_type rank() const
  {
    int r;
    OVXX_MPI_CHECK_RESULT(MPI_Comm_rank, (*this, &r));
    return static_cast<processor_type>(r);
  }
  /// Determine the number of processes in a communicator.
  length_type size() const
  {
    int s;
    OVXX_MPI_CHECK_RESULT(MPI_Comm_size, (*this, &s));
    return static_cast<length_type>(s);
  }
  /// This routine constructs a new group whose members are the
  /// processes within this communicator.
  Group group() const 
  {
    MPI_Group g;
    OVXX_MPI_CHECK_RESULT(MPI_Comm_group, (*this, &g));
    return Group(g);
  }
  pvec_type const& pvec() const { return pvec_;}

  Communicator split(int color, int key) const
  {
    MPI_Comm c;
    OVXX_MPI_CHECK_RESULT(MPI_Comm_split, (*this, color, key, &c));
    return Communicator(c, comm_take_ownership);
  }
  Communicator split(int color) const { return split(color, rank());}

  void barrier() const { MPI_Barrier(*this);}

  template <typename T>
  void buf_send(processor_type dest_proc, T* data, length_type size);

  template <typename T>
  void send(processor_type dest_proc, T* data, length_type size,
	    request_type& req);

  void send(processor_type dest_proc, chain_type& chain, request_type& req);

  template <typename T>
  void recv(processor_type src_proc, T* data, length_type size);

  void recv(processor_type src_proc, chain_type& chain);

  void wait(request_type& req);

  template <typename T>
  void broadcast(processor_type root_proc, T* data, length_type size);

  template <typename T>
  T allreduce(reduction_type rdx, T value);

  int impl_ll_pset() const VSIP_NOTHROW { return 0;}
  // { return ll_pset_type(); }


  friend bool operator==(Communicator const&, Communicator const&);

private:
  std::shared_ptr<MPI_Comm> impl_;
  pvec_type pvec_;
};

inline bool
operator==(Communicator const &comm1, Communicator const &comm2)
{
  int result;
  OVXX_MPI_CHECK_RESULT(MPI_Comm_compare, (comm1, comm2, &result));
  return result == MPI_IDENT;
}

inline bool
operator!=(Communicator const &comm1, Communicator const &comm2)
{
  return !operator==(comm1, comm2);
}

template <typename T>
inline void
Communicator::buf_send(processor_type dest_proc, T *data, length_type size)
{
  OVXX_MPI_CHECK_RESULT(MPI_Bsend, 
    (data, size, Datatype<T>::value(), dest_proc, 0, *this));
}

template <typename T>
inline void
Communicator::send(processor_type dest_proc,
		   T *data,
		   length_type size,
		   request_type &req)
{
  OVXX_MPI_CHECK_RESULT(MPI_Isend, 
    (data, size, Datatype<T>::value(), dest_proc, 0, *this, &req));
}

inline void
Communicator::send(processor_type dest_proc, chain_type &chain, request_type &req)
{
  OVXX_MPI_CHECK_RESULT(MPI_Isend,
    (MPI_BOTTOM, 1, chain, dest_proc, 0, *this, &req));
}

template <typename T>
inline void
Communicator::recv(processor_type src_proc, T *data, length_type size)
{
  MPI_Status status;
  OVXX_MPI_CHECK_RESULT(MPI_Recv, 
    (data, size, Datatype<T>::value(), src_proc, 0, *this, &status));
}

inline void
Communicator::recv(processor_type src_proc, chain_type &chain)
{
  MPI_Status status;
  OVXX_MPI_CHECK_RESULT(MPI_Recv, 
    (MPI_BOTTOM, 1, chain, src_proc, 0, *this, &status));
}

/// Wait for a previous communication (send or receive) to complete.
inline void
Communicator::wait(request_type& req)
{
  MPI_Status status;
  MPI_Wait(&req, &status);
}

/// Broadcast a value from root processor to other processors.
template <typename T>
inline void
Communicator::broadcast(processor_type root_proc, T* data, length_type size)
{
  OVXX_MPI_CHECK_RESULT(MPI_Bcast,
    (data, size, Datatype<T>::value(), root_proc, *this));
}

/// Reduce a value from all processors to all processors.
template <typename T>
inline T
Communicator::allreduce(reduction_type rtype, T value)
{
  T result;

  MPI_Op op;

  switch (rtype)
  {
  case reduce_all_true:		op = MPI_LAND; break;
  case reduce_all_true_bool:	op = MPI_BAND; break;
  case reduce_any_true:		op = MPI_LOR; break;
  case reduce_any_true_bool:	op = MPI_BOR; break;
  case reduce_sum:		op = MPI_SUM; break;
  case reduce_min:		op = MPI_MIN; break;
  case reduce_max:		op = MPI_MAX; break;
  default: OVXX_UNREACHABLE("invalid reduction type");
  }

  OVXX_MPI_CHECK_RESULT(MPI_Allreduce, 
    (&value, &result, 1, Datatype<T>::value(), op, *this));
  return result;
}

} // namespace ovxx::mpi
} // namespace ovxx

#endif
