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
#if defined(OVXX_ENABLE_THREADING)
# include <ovxx/c++11/thread.hpp>
#endif
#if OVXX_HAVE_MPI
# include <ovxx/mpi/service.hpp>
#endif
#if defined(OVXX_ENABLE_OMP)
# include <omp.h>
#endif
#if defined(OVXX_HAVE_CVSIP)
extern "C" {
# include <vsip.h>
}
#endif
#if defined(OVXX_FFTW_THREADS)
# include <fftw3.h>
#endif

using namespace ovxx;

namespace
{
// Part of the library need to be initialized once per application,
// other parts once per thread. Thus we maintain one global and one
// thread-local counter.
// In addition, we initialize OMP threads so OMP pragmas will work
// correctly.
unsigned int global_count = 0;
#if defined(OVXX_ENABLE_THREADING)
mutex global_count_guard;
thread_local unsigned int thread_local_count = 0;
#ifdef OVXX_ENABLE_OMP
thread_local library *omp_library = 0;
#endif
#else
unsigned int thread_local_count = 0;
#endif

void initialize(int &argc, char **&argv)
{
  {
#if defined(OVXX_ENABLE_THREADING)
    lock_guard<mutex> lock(global_count_guard);
#endif
    ++global_count;
  }
  if (thread_local_count++) return;

  if (global_count == 1)
  {
#ifndef OVXX_TIMER_SYSTEM
    cxx11::chrono::high_resolution_clock::init();
#endif
#if (OVXX_HAVE_CVSIP)
    vsip_init(0);
#endif
#if defined(OVXX_FFTW_THREADS)
    int status = 0;
# ifdef OVXX_FFTW_HAVE_FLOAT
    status = fftwf_init_threads();
    if (!status)
      OVXX_DO_THROW(std::runtime_error("Error during FFTW initialization"));
    fftwf_plan_with_nthreads(4);
# endif
# ifdef OVXX_FFTW_HAVE_DOUBLE
    status = fftw_init_threads();
    if (!status)
      OVXX_DO_THROW(std::runtime_error("Error during FFTW initialization"));
    fftw_plan_with_nthreads(4);
# endif
#endif // OVXX_FFTW_THREADS
  }
  if (thread_local_count == 1)
  {
    allocator::initialize(argc, argv);
#if OVXX_HAVE_MPI
    mpi::initialize(argc, argv);
#endif
  }
#ifdef OVXX_ENABLE_OMP
  if (global_count == 1)
  {
    // Initialize OpenMP threads
#pragma omp parallel
    // thread 0 is the main thread, which is already
    // initialized.
    if (!omp_library && omp_get_thread_num() != 0)
      omp_library = new library();
  }
#endif
}

/// Destructor worker function.

/// There is only one destructor, but for symmetry we put the
/// bulk of the finalization logic in this function.
void finalize()
{
#ifdef OVXX_ENABLE_OMP
  // finalize OMP threads from the main thread.
  if (omp_get_thread_num() == 0)
  {
#pragma omp parallel
    delete omp_library;
    omp_library = 0;
  }
#endif
  {
#if OVXX_ENABLE_THREADING
    lock_guard<mutex> lock(global_count_guard);
#endif
    --global_count;
  }
  --thread_local_count;
  if (!thread_local_count)
  {
#if OVXX_HAVE_MPI
    mpi::finalize(global_count == 0);
    allocator::finalize();
#endif
  }
  if (!global_count)
  {
#if (OVXX_HAVE_CVSIP)
    vsip_finalize(0);
#endif
  }
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
