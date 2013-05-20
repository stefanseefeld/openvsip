//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_service_hpp_
#define ovxx_parallel_service_hpp_

#include <ovxx/config.hpp>
#ifdef OVXX_PARALLEL
# include <ovxx/mpi/service.hpp>
#else
# include <ovxx/parallel/serial.hpp>
#endif

#endif
