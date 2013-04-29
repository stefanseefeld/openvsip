//
// Copyright (c) 2009 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include "services.hpp"
#include <vsip/core/argv_utils.hpp>
#include <cstring>

namespace vsip
{
namespace impl
{
namespace mpi
{

#if VSIP_IMPL_ENABLE_THREADING
thread_local Communicator *Service::default_communicator_;
#else
Communicator *Service::default_communicator_;
#endif

Service::Service(int& argc, char**& argv)
  : initialized_(false),
    buf_(new char[BUFSIZE])
{
  int externally_initialized;
  VSIP_IMPL_MPI_CHECK_RESULT(MPI_Initialized, (&externally_initialized));
  if (!externally_initialized)
  {
    initialized_ = MPI_Init(&argc, &argv) == 0;
    assert(initialized_);
  }
  default_communicator_ = new communicator_type(MPI_COMM_WORLD, communicator_type::comm_attach);
  MPI_Buffer_attach(buf_, BUFSIZE);
  
  // Initialize complex datatypes.
  Datatype<std::complex<float> >::value();
  Datatype<std::complex<double> >::value();
}

Service::~Service()
{
  delete default_communicator_;
  if (initialized_)
    MPI_Finalize();
  delete[] buf_;
}

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip
