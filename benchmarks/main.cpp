/* Copyright (c) 2005, 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark main.

#include <fstream>
#include <iostream>
#if defined(_MC_EXEC)
#  include <unistd.h>
#endif

#include <vsip/initfin.hpp>
#include <vsip/core/check_config.hpp>
#include <vsip/core/huge_page_pool.hpp>

#include "benchmarks.hpp"

#if !_WIN32
# include <sys/types.h>
# include <unistd.h>
#endif

using namespace vsip;

extern int test(Loop1P& loop, int what);
extern void defaults(Loop1P& loop);



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  if (!vsip_csl::profile::DefaultTime::valid)
  {
    std::cerr << argv[0] << ": timer "
	      << vsip_csl::profile::DefaultTime::name()
	      << " not valid for benchmarking"
	      << std::endl;
    exit(-1);
  }

  Loop1P loop;
  bool   verbose = false;
  bool   pause   = false;

  loop.goal_sec_ = 0.25;
  defaults(loop);

  int what = 0;

  for (int i=1; i<argc; ++i)
  {
    if (sscanf(argv[i], "-%d", &what))
      ;
    else if (!strcmp(argv[i], "-start"))
    {
      loop.start_ = atoi(argv[++i]);
      loop.cal_   = loop.start_;
    }
    else if (!strcmp(argv[i], "-stop"))
      loop.stop_ = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-cal"))
      loop.cal_ = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-loop_start"))
      loop.loop_start_ = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-fix_loop"))
      loop.fix_loop_ = true;
    else if (!strcmp(argv[i], "-samples"))
      loop.samples_ = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-ms"))
      loop.goal_sec_ = (double)atoi(argv[++i])/100;
    else if (!strcmp(argv[i], "-verbose"))
      verbose = true;
    else if (!strcmp(argv[i], "-pause"))
      pause = true;
    else if (!strcmp(argv[i], "-param"))
      loop.user_param_ = atoi(argv[++i]);
    else if (!strncmp(argv[i], "-p:", 3))
    {
      std::string name(argv[i]+3);
      std::string value(argv[++i]);
      loop.param_[name] = value;
    }
    else if (!strcmp(argv[i], "-ops"))
      loop.metric_ = ops_per_sec;
    else if (!strcmp(argv[i], "-pts"))
      loop.metric_ = pts_per_sec;
    else if (!strcmp(argv[i], "-iob"))
      loop.metric_ = iob_per_sec;
    else if (!strcmp(argv[i], "-wiob"))
      loop.metric_ = wiob_per_sec;
    else if (!strcmp(argv[i], "-riob"))
      loop.metric_ = riob_per_sec;
    else if (!strcmp(argv[i], "-all"))
      loop.metric_ = all_per_sec;
    else if (!strcmp(argv[i], "-data"))
      loop.metric_ = data_per_sec;
    else if (!strcmp(argv[i], "-lat"))
      loop.metric_ = secs_per_pt;
    else if (!strcmp(argv[i], "-geom"))
      loop.progression_ = geometric;
    else if (!strcmp(argv[i], "-linear"))
    {
      loop.progression_ = linear;
      loop.prog_scale_ = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-center"))
    {
      loop.range_ = centered_range;
      loop.center_ = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-mfile"))
    {
      loop.progression_ = userfile;
      char const* filename = argv[++i];
      std::ifstream file(filename);
      unsigned value;
      while (file >> value)
	loop.m_array_.push_back(value);
      loop.cal_   = 0;
      loop.start_ = 0;
      loop.stop_  = loop.m_array_.size()-1;
    }
    else if (!strcmp(argv[i], "-mem"))
      loop.lhs_ = lhs_mem;
    else if (!strcmp(argv[i], "-prof"))
      loop.do_prof_  = true;
    else if (!strcmp(argv[i], "-show_loop"))
      loop.show_loop_ = true;
    else if (!strcmp(argv[i], "-show_time"))
      loop.show_time_ = true;
    else if (!strcmp(argv[i], "-steady"))
    {
      loop.mode_  = steady_mode;
      loop.start_ = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-diag"))
      loop.mode_ = diag_mode;
    else if (!strcmp(argv[i], "-nocal"))
      loop.do_cal_ = false;
    else if (!strcmp(argv[i], "-single"))
    {
      loop.mode_ = single_mode;
      loop.cal_  = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-lib_config"))
    {
      std::cout << vsip::impl::library_config();
      return 0;
    }
    else if (!strcmp(argv[i], "-pool"))
    {
      ++i;
      if (!strcmp(argv[i], "def"))
	;
#if VSIP_IMPL_ENABLE_HUGE_PAGE_POOL
      else if (!strcmp(argv[i], "huge"))
	loop.pool_ = new vsip::impl::Huge_page_pool("/huge/benchmark.bin", 9);
      else if (!strncmp(argv[i], "huge:", 5))
      {
	int pages = atoi(argv[i]+5);
	loop.pool_ = new vsip::impl::Huge_page_pool("/huge/benchmark.bin",
						    pages);
      }
#endif
      else
	std::cerr << "ERROR: Unknown pool type: " << argv[i] << std::endl;
    }
    else
      std::cerr << "ERROR: Unknown argument: " << argv[i] << std::endl;
  }

  if (loop.metric_ == data_per_sec && loop.samples_ < 3)
  {
    std::cerr << "ERROR: -samples must be >= 3 when using -data" << std::endl;
    exit(-1);
  }

  if (verbose)
  {
    std::cout << "what = " << what << std::endl;
    std::cout << "sec  = " << loop.goal_sec_ << std::endl;
  }

  if (pause)
  {
    // Enable this section for easier debugging.
    impl::Communicator& comm = impl::default_communicator();
#if !_WIN32
    pid_t pid = getpid();
#endif

    std::cout << "rank: "   << comm.rank()
	      << "  size: " << comm.size()
#if !_WIN32
	      << "  pid: "  << pid
#endif
	      << std::endl;

    // Stop each process, allow debugger to be attached.
    if (comm.rank() == 0) getchar();
    comm.barrier();
  }

  loop.what_ = what;

  test(loop, what);
}

