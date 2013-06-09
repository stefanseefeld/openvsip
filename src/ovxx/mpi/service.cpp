//
// Copyright (c) 2009 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "service.hpp"
#include <cstring>
#include <iostream>

namespace
{
#if OVXX_ENABLE_THREADING
thread_local ovxx::mpi::Communicator *communicator;
#else
ovxx::mpi::Communicator *communicator;
#endif
vsip::length_type const BUFSIZE = 2*256*256*4;
char *buffer = 0;
bool finalize_mpi = false;
}

namespace ovxx
{
namespace mpi
{

void initialize(int &argc, char **&argv)
{
  buffer = new char[BUFSIZE];
  int initialized;
  OVXX_MPI_CHECK_RESULT(MPI_Initialized, (&initialized));
  if (!initialized)
  {
    initialized = MPI_Init(&argc, &argv) == 0;
    OVXX_INVARIANT(initialized);
    finalize_mpi = true;
  }
  communicator = new Communicator(MPI_COMM_WORLD, Communicator::comm_attach);
  MPI_Buffer_attach(buffer, BUFSIZE);
  
  // Initialize complex datatypes.
  Datatype<complex<float> >::value();
  Datatype<complex<double> >::value();  
}
void finalize()
{
  delete communicator;
  communicator = 0;
  if (finalize_mpi)
    MPI_Finalize();
  delete[] buffer;
}

} // namespace ovxx::mpi

namespace parallel
{
Communicator &default_communicator()
{
  return *communicator;
}
Communicator set_default_communicator(Communicator c)
{
  Communicator old = *communicator;
  *communicator = c;
  return old;
}
} // namespace ovxx::parallel
} // namespace ovxx

namespace vsip
{
length_type num_processors() VSIP_NOTHROW
{
  return ovxx::parallel::default_communicator().size();
}

processor_type local_processor() VSIP_NOTHROW
{
  return ovxx::parallel::default_communicator().rank();
}

index_type local_processor_index() VSIP_NOTHROW
{
  ovxx::mpi::Communicator::pvec_type const &pvec = 
    ovxx::parallel::default_communicator().pvec(); 
  processor_type proc = local_processor();
  for (index_type i = 0; i != pvec.size(); ++i)
    if (pvec[i] == proc) return i;
  assert(0); // Invalid local processor
}
}
