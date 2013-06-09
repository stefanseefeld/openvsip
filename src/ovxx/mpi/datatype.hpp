//
// Copyright (c) 2010 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_mpi_datatype_hpp_
#define ovxx_mpi_datatype_hpp_

#include <ovxx/support.hpp>
#include <vsip/domain.hpp>
#include <mpi.h>

namespace ovxx
{
namespace mpi
{

/// Provide a mapping from C++ types to MPI types.
template <typename T> struct Datatype;

#define OVXX_DATATYPE(CTYPE, MPITYPE)			\
template <>			       			\
struct Datatype<CTYPE>		       			\
{					       		\
  static MPI_Datatype value() { return MPITYPE;}       	\
};

OVXX_DATATYPE(char,           MPI_CHAR)
OVXX_DATATYPE(short,          MPI_SHORT)
OVXX_DATATYPE(int,            MPI_INT)
OVXX_DATATYPE(long,           MPI_LONG)
OVXX_DATATYPE(signed char,    MPI_CHAR)
OVXX_DATATYPE(unsigned char,  MPI_UNSIGNED_CHAR)
OVXX_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT)
OVXX_DATATYPE(unsigned int,   MPI_UNSIGNED)
OVXX_DATATYPE(unsigned long,  MPI_UNSIGNED_LONG)
OVXX_DATATYPE(float,          MPI_FLOAT)
OVXX_DATATYPE(double,         MPI_DOUBLE)
OVXX_DATATYPE(long double,    MPI_LONG_DOUBLE)

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

#undef OVXX_DATATYPE

} // namespace ovxx::mpi
} // namespace ovxx

#endif
