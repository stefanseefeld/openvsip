//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/library.hpp>
#include <ovxx/allocator.hpp>
#include <ovxx/c++11/chrono.hpp>
#if OVXX_HAVE_MPI
# include <ovxx/mpi/service.hpp>
#endif
#if defined(OVXX_HAVE_CVSIP)
extern "C" {
# include <vsip.h>
}
#endif
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cassert>

using namespace ovxx;

namespace
{
unsigned int use_count = 0;

void initialize(int &argc, char **&argv)
{
  if (use_count++ != 0) return;

  allocator::initialize(argc, argv);
#ifndef OVXX_TIMER_SYSTEM
  cxx11::chrono::high_resolution_clock::init();
#endif
#if (OVXX_HAVE_CVSIP)
  vsip_init(0);
#endif
#if OVXX_HAVE_MPI
  mpi::initialize(argc, argv);
#endif
}

/// Destructor worker function.

/// There is only one destructor, but for symmetry we put the
/// bulk of the finalization logic in this function.
void finalize()
{
  if (--use_count != 0) return;

#if OVXX_HAVE_MPI
  mpi::finalize();
#endif
#if (OVXX_HAVE_CVSIP)
  vsip_finalize(0);
#endif
  allocator::finalize();
}
} // namespace <unnamed>

namespace ovxx
{
library::library()
{
  // Fake argc and argv.  MPICH-1 (1.2.6) expects a program name to
  // be provided and will segfault otherwise.

  int argc = 1;
  char *argv_storage[2];
  char **argv = argv_storage;

  argv[0] = (char*) "program-name";
  argv[1] = NULL;

  initialize(argc, argv);
}

library::library(int &argc, char **&argv)
{
  initialize(argc, argv);
}

library::~library()
{
  finalize();
}

} // namespace ovxx
