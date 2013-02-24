/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

#ifndef VSIP_CORE_PARALLEL_MPI_HPP
#define VSIP_CORE_PARALLEL_MPI_HPP

// Only one parallel/xxx.hpp header should be included
#ifdef VSIP_IMPL_PAR_SERVICES_UNIQUE
#  error "Only one parallel/xxx.hpp should be included"
#endif
#define VSIP_IMPL_PAR_SERVICES_UNIQUE

#include <vector>
#include <complex>

// Place "<" ">" brackets around the mpi header:
#define VSIP_IMPL_MPI_H_BRACKET < VSIP_IMPL_MPI_H >

#include <vsip/core/config.hpp>
#if VSIP_IMPL_MPI_H_TYPE == 1
#  include <mpi.h>
#elif VSIP_IMPL_MPI_H_TYPE == 2
#  include <mpi/mpi.h>
#endif
#include <vsip/support.hpp>
#include <vsip/core/reductions/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/parallel/assign_fwd.hpp>

namespace vsip
{
namespace impl
{

typedef int par_ll_pbuf_type;
typedef int par_ll_pset_type;

namespace mpi
{

/// Traits class to determine MPI_DATATYPE from a C++ datatype

template <typename T>
struct Mpi_datatype;

#define VSIP_IMPL_MPIDATATYPE(CTYPE, MPITYPE)				\
template <>								\
struct Mpi_datatype<CTYPE>						\
{									\
  static MPI_Datatype value() { return MPITYPE; }			\
};

VSIP_IMPL_MPIDATATYPE(char,           MPI_CHAR)
VSIP_IMPL_MPIDATATYPE(short,          MPI_SHORT)
VSIP_IMPL_MPIDATATYPE(int,            MPI_INT)
VSIP_IMPL_MPIDATATYPE(long,           MPI_LONG)
VSIP_IMPL_MPIDATATYPE(signed char,    MPI_CHAR)
VSIP_IMPL_MPIDATATYPE(unsigned char,  MPI_UNSIGNED_CHAR)
VSIP_IMPL_MPIDATATYPE(unsigned short, MPI_UNSIGNED_SHORT)
VSIP_IMPL_MPIDATATYPE(unsigned int,   MPI_UNSIGNED)
VSIP_IMPL_MPIDATATYPE(unsigned long,  MPI_UNSIGNED_LONG)
VSIP_IMPL_MPIDATATYPE(float,          MPI_FLOAT)
VSIP_IMPL_MPIDATATYPE(double,         MPI_DOUBLE)
VSIP_IMPL_MPIDATATYPE(long double,    MPI_LONG_DOUBLE)

template <dimension_type dim>
struct Mpi_datatype<Index<dim> >
{
  static MPI_Datatype value()
  {
    static bool         first = true;
    static MPI_Datatype datatype;

    if (first)
    {
      first = false;
      MPI_Type_contiguous(dim, Mpi_datatype<index_type>::value(), &datatype);
      MPI_Type_commit(&datatype);
    }

    return datatype;
  }
};

template <typename T>
struct Mpi_datatype<std::complex<T> >
{
  static MPI_Datatype value()
  {
    static bool         first = true;
    static MPI_Datatype datatype;

    if (first)
    {
      first = false;
      MPI_Type_contiguous(2, Mpi_datatype<T>::value(), &datatype);
      MPI_Type_commit(&datatype);
    }

    return datatype;
  }
};

template <>
struct Mpi_datatype<bool>
{
  static MPI_Datatype value()
  {
    static bool         first = true;
    static MPI_Datatype datatype;

    if (first)
    {
      first = false;

      if (sizeof(bool) == 1)
	datatype = MPI_BYTE;
      else
      {
	MPI_Type_contiguous(sizeof(bool), MPI_BYTE, &datatype);
	MPI_Type_commit(&datatype);
      }
    }

    return datatype;
  }
};

#undef VSIP_IMPL_MPIDATATYPE



/// Return the processor's rank in an MPI_Communicator.
processor_type get_rank(MPI_Comm comm);

/// Return the size (number of processors) of an MPI_Communicator.
length_type    get_size(MPI_Comm comm);

inline void
chk(int val) { assert(val == MPI_SUCCESS); }




/// DMA Chain
class Chain_builder
{
public:
  Chain_builder()
    : lengths_ (),
      offsets_ (),
      types_   (),
      alltypes_()
  {}

  ~Chain_builder()
  {
    for (unsigned i=0; i<alltypes_.size(); ++i)
      MPI_Type_free(&alltypes_[i]);
  }

  template <typename T>
  void add(ptrdiff_t offset, int stride, unsigned length)
  {
    MPI_Datatype vtype;
    chk(MPI_Type_vector(length, 1, stride, Mpi_datatype<T>::value(), &vtype));
    chk(MPI_Type_commit(&vtype));

    lengths_.push_back(1);
    offsets_.push_back(offset);
    types_.push_back(vtype);

    alltypes_.push_back(vtype);
  }

  template <typename T>
  void add(ptrdiff_t offset,
	   int stride0, unsigned length0,
	   int stride1, unsigned length1)
  {
#if 0
    // This doesn't work because MPI gets confused about the extent of
    // vtyp0.  If vtype has a stride greater then vtype1  with the intent
    // that vtype1 will interleave multiple vtype0's, MPI thinks the
    // intent is for the vtype0s to be contiguous.
    // Possible to work around this with explicit upper/lower bound.
    MPI_Datatype vtype0;
    MPI_Datatype vtype1;

    chk(MPI_Type_vector(length0, 1, stride0, Mpi_datatype<T>::value(), &vtype0));
    chk(MPI_Type_commit(&vtype0));
    chk(MPI_Type_hvector(length1, 1, stride1*sizeof(T), vtype0, &vtype1));
    chk(MPI_Type_commit(&vtype1));

    MPI_Aint addr;
    chk(MPI_Address(offset, &addr));

    lengths_.push_back(1);
    offsets_.push_back(addr);
    types_.push_back(vtype1);

    alltypes_.push_back(vtype1);
    alltypes_.push_back(vtype0);
#else
    MPI_Datatype vtype0;
    chk(MPI_Type_vector(length1, 1, stride1, Mpi_datatype<T>::value(), &vtype0));
    chk(MPI_Type_commit(&vtype0));

    for (unsigned i=0; i<length0; ++i)
    {
      lengths_.push_back(1);
      offsets_.push_back(offset + sizeof(T)*i*stride0);
      types_.push_back(vtype0);
    }
    alltypes_.push_back(vtype0);
#endif
  }

  void* base() { return MPI_BOTTOM; }

  MPI_Datatype get_chain()
  {
    MPI_Datatype type;

    MPI_Type_struct(
      lengths_.size(),
      &lengths_[0],
      &offsets_[0],
      &types_[0],
      &type);

    MPI_Type_commit(&type);
    return type;
  }

  void stitch(void* base, MPI_Datatype chain)
  {
    MPI_Aint addr;
    chk(MPI_Address(base, &addr));

    lengths_.push_back(1);
    offsets_.push_back(addr);
    types_.push_back(chain);
  }

  void stitch(std::pair<void*,void*> base, MPI_Datatype chain)
  {
    stitch(base.first, chain);
    stitch(base.second, chain);
  }

  bool is_empty() const { return (lengths_.size() == 0); }

  // Private member data.
private:
  std::vector<int>		lengths_;
  std::vector<MPI_Aint>		offsets_;
  std::vector<MPI_Datatype>	types_;

  std::vector<MPI_Datatype>	alltypes_;
  MPI_Datatype datatype_;
};



/// Communicator class for MPI.

/// A VSIPL++ Communicator is essentially just an MPI Communicator at
/// the moment.

class Communicator
{
public:
  typedef MPI_Request  request_type;
  typedef MPI_Datatype chain_type;
  typedef std::vector<processor_type> pvec_type;

public:
  Communicator()
    : comm_(MPI_COMM_NULL), rank_(0), size_(0), pvec_(0)
  {}

  Communicator(MPI_Comm comm)
    : comm_(comm),
      rank_(get_rank(comm_)),
      size_(get_size(comm_)),
      pvec_(size_)
  {
    for (index_type i=0; i<size_; ++i)
    {
      pvec_[i] = static_cast<processor_type>(i);
    }
  }

  processor_type   rank() const { return rank_; }
  length_type      size() const { return size_; }
  pvec_type const& pvec() const { return pvec_; }

  void barrier() const { MPI_Barrier(comm_); }

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

  par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return par_ll_pset_type(); }

  friend bool operator==(Communicator const&, Communicator const&);

private:
  MPI_Comm		 comm_;
  processor_type	 rank_;
  length_type		 size_;
  pvec_type		 pvec_;
};

} // namespace vsip::impl::mpi



typedef mpi::Communicator Communicator;
typedef mpi::Chain_builder Chain_builder;
typedef Chained_assign par_assign_impl_type;

inline void
create_ll_pset(
  std::vector<processor_type> const&,
  par_ll_pset_type&)
{}

inline void
destroy_ll_pset(par_ll_pset_type&)
{}

inline void
free_chain(MPI_Datatype chain)
{
  MPI_Type_free(&chain);
}



/// Par_service class for MPI.

class Par_service
{
  // Compile-time values and typedefs.
public:
  typedef mpi::Communicator communicator_type;

  // Constructors.
public:
  Par_service(int& argc, char**& argv);
  ~Par_service();

  static communicator_type& default_communicator()
  {
    return default_communicator_;
  }

private:
  static length_type const BUFSIZE = 2*256*256*4;
  static communicator_type default_communicator_;

  bool  initialized_;
  char *buf_;
};

// supported reductions

template <reduction_type rtype,
	  typename       T>
struct Reduction_supported
{ static bool const value = false; };

template <> struct Reduction_supported<reduce_sum, int>
{ static bool const value = true; };
template <> struct Reduction_supported<reduce_sum, float>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_all_true, int>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_all_true_bool, bool>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_any_true, int>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_any_true_bool, bool>
{ static bool const value = true; };

namespace mpi
{

inline processor_type
get_rank(MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return static_cast<processor_type>(rank);
}



inline length_type
get_size(MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  return static_cast<length_type>(size);
}



inline bool
operator==(Communicator const& comm1, Communicator const& comm2)
{
  return (comm1.comm_ == comm2.comm_);
}



inline bool
operator!=(Communicator const& comm1, Communicator const& comm2)
{
  return !operator==(comm1, comm2);
}



template <typename T>
inline void
Communicator::buf_send(
  processor_type dest_proc,
  T*             data,
  length_type    size)
{
  int ierr = MPI_Bsend(data, size, Mpi_datatype<T>::value(),
		       dest_proc, 0, comm_);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));
}



template <typename T>
inline void
Communicator::send(
  processor_type dest_proc,
  T*             data,
  length_type    size,
  request_type&  req)
{
  int ierr = MPI_Isend(data, size, Mpi_datatype<T>::value(),
		       dest_proc, 0, comm_,
		       &req);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));
}


inline void
Communicator::send(
  processor_type dest_proc,
  chain_type&    chain,
  request_type&  req)
{
  int ierr = MPI_Isend(MPI_BOTTOM, 1, chain, dest_proc, 0, comm_, &req);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));
}



template <typename T>
inline void
Communicator::recv(
  processor_type   src_proc,
  T*               data,
  length_type      size)
{
  MPI_Status status;

  int ierr = MPI_Recv(data, size, Mpi_datatype<T>::value(), src_proc, 0,
		      comm_, &status);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));
}



inline void
Communicator::recv(
  processor_type   src_proc,
  chain_type&      chain)
{
  MPI_Status status;

  int ierr = MPI_Recv(MPI_BOTTOM, 1, chain, src_proc, 0, comm_, &status);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));
}



/// Wait for a previous communication (send or receive) to complete.

inline void
Communicator::wait(
  request_type& req)
{
  MPI_Status status;
  MPI_Wait(&req, &status);
}



/// Broadcast a value from root processor to other processors.

template <typename T>
inline void
Communicator::broadcast(processor_type root_proc, T* data, length_type size)
{
  int ierr = MPI_Bcast(data, size, Mpi_datatype<T>::value(), root_proc,
		       comm_);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));
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
  default: assert(false);
  }

  int ierr = MPI_Allreduce(&value, &result, 1, Mpi_datatype<T>::value(),
		       op, comm_);
  if (ierr != MPI_SUCCESS)
    VSIP_IMPL_THROW(impl::unimplemented("MPI error handling."));

  return result;
}

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_PAR_SERVICES_MPI_HPP
