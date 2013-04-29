/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#ifndef vsip_core_mpi_chain_builder_hpp_
#define vsip_core_mpi_chain_builder_hpp_

#include <vsip/support.hpp>
#include <vsip/core/mpi/exception.hpp>
#include <vsip/core/mpi/datatype.hpp>
#include <vsip/core/mpi/group.hpp>
#include <mpi.h>

namespace vsip
{
namespace impl
{
namespace mpi
{

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
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_vector,
      (length, 1, stride, Datatype<T>::value(), &vtype));
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_commit, (&vtype));

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

    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_vector, 
      (length0, 1, stride0, Datatype<T>::value(), &vtype0));
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_commit, (&vtype0));
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_hvector,
      (length1, 1, stride1*sizeof(T), vtype0, &vtype1));
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_commit, (&vtype1));

    MPI_Aint addr;
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Address, (offset, &addr));

    lengths_.push_back(1);
    offsets_.push_back(addr);
    types_.push_back(vtype1);

    alltypes_.push_back(vtype1);
    alltypes_.push_back(vtype0);
#else
    MPI_Datatype vtype0;
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_vector,
      (length1, 1, stride1, Datatype<T>::value(), &vtype0));
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Type_commit, (&vtype0));

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
    VSIP_IMPL_MPI_CHECK_RESULT(MPI_Address, (base, &addr));

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


inline void
free_chain(MPI_Datatype chain)
{
  MPI_Type_free(&chain);
}

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip

#endif
