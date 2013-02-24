/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/vsip_csl/load_view.hpp
    @author  Jules Bergmann
    @date    2006-09-28
    @brief   VSIPL++ Library: Unit-tests for vsip_csl/load_view.hpp
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define DEBUG 0

#include <iostream>
#if DEBUG
#include <unistd.h>
#endif
#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-storage.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/save_view.hpp>

#include "load_save.hpp"
#include "test_common.hpp"

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

#if DEBUG
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
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

  // Note: Complex versions of these tests are found in the module
  // 'load_view_cplx.cpp'.  The tests were split to improve compile time.
  test_type<int>();
  test_type<float>();
  test_type<double>();
}


