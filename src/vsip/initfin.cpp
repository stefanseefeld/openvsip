/* Copyright (c) 2005 CodeSourcery, LLC.  All rights reserved.  */

/** @file    initfin.cpp
    @author  Zack Weinberg
    @date    2005-01-19
    @brief   VSIPL++ Library: [initfin] Initialization and finalization
             (implementation).  */

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/memory_pool.hpp>
#if defined(VSIP_IMPL_CBE_SDK) && !defined(VSIP_IMPL_REF_IMPL)
# include <vsip/opt/cbe/ppu/task_manager.hpp>
#endif
#if defined(VSIP_IMPL_NUMA) && !defined(VSIP_IMPL_REF_IMPL)
# include <vsip/opt/numa.hpp>
#endif
#if defined(VSIP_IMPL_HAVE_CUDA) && !defined(VSIP_IMPL_REF_IMPL)
# include <vsip/opt/cuda/library.hpp>
#endif
#include <cstring>

using namespace vsip;

/***********************************************************************
  Definitions
***********************************************************************/

/// Library initialization and finalization happen only once,
/// no matter how many \c vsipl objects are created.
impl::Checked_counter vsipl::use_count = 0;

namespace
{
impl::Par_service *par_service = 0;
#ifndef VSIP_IMPL_REF_IMPL
impl::profile::Profiler_options *profiler_opts = 0;
#endif
}

/// If there are no other extant \c vsipl objects, this function
/// will initialize the library so it can be used.  If other
/// \c vsipl objects exist, this function returns immediately.
///
/// Typical usage:
/// \code
/// int
/// main()
/// {
///   vsip::vsipl v;
///
///   // Use library.
/// }
/// \endcode
vsipl::vsipl()
{
  // Fake argc and argv.  MPICH-1 (1.2.6) expects a program name to
  // be provided and will segfault otherwise.

  int argc = 1;
  char*  argv_storage[2];
  char** argv = argv_storage;

  argv[0] = (char*) "program-name";
  argv[1] = NULL;

  initialize_library(argc, argv);
}

///   @param argc Reference to command-line argument count.
///   @param argv Reference to command-line argument vector.
///
/// Like the default constructor, except that the provided
/// command-line argument vector will be inspected for options
/// which modify the library's behavior.  These arguments will
/// be removed from \p argv, and \p argc adjusted to match.
///
/// Typical usage:
/// \code
/// int
/// main(int argc, char **argv)
/// {
///   vsip::vsipl v(argc, argv);
///
///   // Use library.
/// }
/// \endcode
///
/// Future reference: The specification is silent on what options
/// there might be that modify the library's behavior, but I think
/// it would be wise to lay some ground rules.  My suggestions:
///
///    - The library recognizes its special options no matter where
///      they appear on the command line, and will splice out its
///      special options even if they appear immediately after an
///      option to the program that takes a parameter.  (This is
///      because we have no way of knowing which of the program's
///      command line arguments take separate-argument parameters.)
///
///    - The library will not examine the command line past the first
///      occurrence of "--" as a complete argument.  (This is a
///      standard Unix convention for argument processing.)
///
///    - All special library options begin with the string "--vsip-".
///
///    - No special library option takes a parameter as a separate
///      command line argument.  Options which require parameters
///      are written in the form "--vsip-OPTION=PARAMETER".

vsipl::vsipl(int& argc, char**& argv)
{
  initialize_library(argc, argv);
}

/// If other \c vsipl objects exist, this function returns
/// immediately.  If this is the last \c vsipl object, all data
/// structures created by the first creation of a \c vsipl object
/// are destroyed.  The library may not be used thereafter, unless
/// a new \c vsipl object is created.
///
/// This function is not normally called explicitly.
vsipl::~vsipl() VSIP_NOTHROW
{
  finalize_library();
}

///   @param argc Reference to command-line argument count.
///   @param argv Reference to command-line argument vector.
///
/// The bulk of the initialization logic is here so that
/// it can be shared by both the constructor variants.
void
vsipl::initialize_library(int& argc, char**& argv)
{
  if (use_count++ != 0)
    return;

  int    use_argc;
  char** use_argv;

  char*  env = getenv("VSIP_OPT");
  bool   argv_is_tmp;
  char const* env_marker = "--vsip-#*%$-env-args-marker";

  if (env)
  {
    char* p=env;
    int env_argc = 0;

    // Skip over inital spaces.
    while( isspace(*env) && *env) ++env;

    // Count the number of arguments from the environment.
    p = env;
    while (*p)
    {
      // skip over arg
      while(!isspace(*p) && *p) ++p;
      env_argc++;
      // skip over space after arg
      while( isspace(*p) && *p) ++p;
    }

    // Allocate a new argv and copy the existing arguments.
    use_argc = argc + env_argc + 1;
    use_argv = new char*[use_argc];

    int i=0;
    for (; i<argc; ++i)
      use_argv[i] = argv[i];

    // Insert argument to track start of env args.  This prevents
    // arguments from environment from propogating back to the application
    // if they aren't processed by VSIPL++.
    use_argv[i++] = const_cast<char*>(env_marker);
      
    // Copy the environment arguments.
    p = env;
    while (*p)
    {
      assert(i < use_argc);
      use_argv[i++] = p;
      // skip over arg
      while(!isspace(*p) && *p) ++p;

      // put 0 after arg
      if (isspace(*p))
	*p++ = 0;

      // skip over additional space after arg
      while( isspace(*p) && *p) ++p;
    }
    argv_is_tmp = true;
  }
  else
  {
    use_argc = argc;
    use_argv = argv;
    argv_is_tmp = false;
  }

#ifndef VSIP_IMPL_REF_IMPL
  // Profiler options are removed as they are processed.  The
  // remaining options are left intact.
  profiler_opts = new impl::profile::Profiler_options(use_argc, use_argv);

# if defined(VSIP_IMPL_NUMA)
  impl::numa::initialize(use_argc, use_argv);
# endif
# if defined(VSIP_IMPL_CBE_SDK)
  impl::cbe::Task_manager::initialize(use_argc, use_argv);
# endif
# if defined(VSIP_IMPL_HAVE_CUDA)
  impl::cuda::initialize(use_argc, use_argv);
# endif

#endif

  par_service = new impl::Par_service(use_argc, use_argv);

  impl::initialize_default_pool(use_argc, use_argv);

  // Copy argv back if necessary
  if (argv_is_tmp)
  {
    int i;
    for (i=0; i<use_argc && i<argc && strcmp(use_argv[i], env_marker); ++i)
      argv[i] = use_argv[i];

    // update use_argc if some of the environment args were not processed.
    use_argc = i;

    for (; i<argc; ++i)
      argv[i] = 0;

    delete[] use_argv;
  }
  argc = use_argc;
}

/// Destructor worker function.

/// There is only one destructor, but for symmetry we put the
/// bulk of the finalization logic in this function.
void
vsipl::finalize_library()
{
  if (--use_count != 0)
    return;

  delete par_service;
  par_service = 0;

#ifndef VSIP_IMPL_REF_IMPL

# if defined(VSIP_IMPL_CBE_SDK)
  impl::cbe::Task_manager::finalize();
# endif
# if defined(VSIP_IMPL_HAVE_CUDA)
  impl::cuda::finalize();
# endif

  delete profiler_opts;
  profiler_opts = 0;

  impl::finalize_default_pool();
#endif
}
