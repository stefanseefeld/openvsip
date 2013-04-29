/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/convolution/dist.cpp
    @author  Jules Bergmann
    @date    2008-08-27
    @brief   VSIPL++ Library: Unit tests for [signal.convolution] items.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/initfin.hpp>
#include <vsip/random.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/metaprogramming.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/ref_conv.hpp>
#include <vsip_csl/error_db.hpp>

#include "convolution.hpp"

#if VERBOSE
#  include <iostream>
#  include <vsip_csl/output.hpp>
#endif

using namespace std;
using namespace vsip;
using namespace vsip_csl;



/***********************************************************************
  Definitions
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if 0
  // Enable this section for easier debugging.
  vsip::impl::Communicator comm = vsip::impl::default_communicator();
  pid_t pid = getpid();

  cout << "rank: "   << comm.rank()
       << "  size: " << comm.size()
       << "  pid: "  << pid
       << endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  cout << "start\n";
#endif

  // Test distributed arguments.
  cases_conv_dist<float>(32, 8, 1);

  return 0;
}
