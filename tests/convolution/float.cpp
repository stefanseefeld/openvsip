//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#include <vsip/vector.hpp>
#include <vsip/signal.hpp>
#include <vsip/initfin.hpp>
#include <vsip/random.hpp>
#include <vsip/parallel.hpp>
#include "convolution.hpp"

int main(int argc, char** argv)
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

  // General tests.
  bool rand = true;
  cases<float>(rand);

  return 0;
}
