/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#ifndef vsip_core_mpi_datatype_hpp_
#define vsip_core_mpi_datatype_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <mpi.h>

namespace vsip
{
namespace impl
{
namespace mpi
{
/// Provide a mapping from C++ types to MPI types.
template <typename T> struct Datatype;

#define VSIP_IMPL_DATATYPE(CTYPE, MPITYPE)		\
template <>			       			\
struct Datatype<CTYPE>		       			\
{					       		\
  static MPI_Datatype value() { return MPITYPE;}       	\
};

VSIP_IMPL_DATATYPE(char,           MPI_CHAR)
VSIP_IMPL_DATATYPE(short,          MPI_SHORT)
VSIP_IMPL_DATATYPE(int,            MPI_INT)
VSIP_IMPL_DATATYPE(long,           MPI_LONG)
VSIP_IMPL_DATATYPE(signed char,    MPI_CHAR)
VSIP_IMPL_DATATYPE(unsigned char,  MPI_UNSIGNED_CHAR)
VSIP_IMPL_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT)
VSIP_IMPL_DATATYPE(unsigned int,   MPI_UNSIGNED)
VSIP_IMPL_DATATYPE(unsigned long,  MPI_UNSIGNED_LONG)
VSIP_IMPL_DATATYPE(float,          MPI_FLOAT)
VSIP_IMPL_DATATYPE(double,         MPI_DOUBLE)
VSIP_IMPL_DATATYPE(long double,    MPI_LONG_DOUBLE)

template <dimension_type D>
struct Datatype<Index<D> >
{
  static MPI_Datatype value()
  {
    static bool first = true;
    static MPI_Datatype datatype;

    if (first)
    {
      first = false;
      MPI_Type_contiguous(D, Datatype<index_type>::value(), &datatype);
      MPI_Type_commit(&datatype);
    }

    return datatype;
  }
};

template <typename T>
struct Datatype<std::complex<T> >
{
  static MPI_Datatype value()
  {
    static bool first = true;
    static MPI_Datatype datatype;

    if (first)
    {
      first = false;
      MPI_Type_contiguous(2, Datatype<T>::value(), &datatype);
      MPI_Type_commit(&datatype);
    }

    return datatype;
  }
};

template <>
struct Datatype<bool>
{
  static MPI_Datatype value()
  {
    static bool first = true;
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

#undef VSIP_IMPL_DATATYPE

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip

#endif
