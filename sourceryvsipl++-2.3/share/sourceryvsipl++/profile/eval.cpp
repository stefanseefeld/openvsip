/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   Profile expression evaluation.

// Defining this will turn on profiling support 
// for the 'dispatch' category.
#define VSIP_PROFILE_DISPATCH 1

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/profile.hpp>

using namespace vsip;
using namespace vsip_csl;

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  // Set tracing mode, and print to stdout.
  // This is equivalent to invoking the application with '--vsip-profile-mode=trace'
  profile::Profile profile("-", profile::pm_trace);
  Vector<float> a(8), b(8);
  Vector<float> c = a * b;
  Vector<float> d = a * b + c;
  return 0;
}
