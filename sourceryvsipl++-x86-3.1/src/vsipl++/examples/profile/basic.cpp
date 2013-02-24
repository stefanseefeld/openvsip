/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description: VSIPL++ Library: Basic User Profiling
///
/// This example illustrates the basics of using user-defined
/// profiling to trace a program's execution and time operations.  Run
/// the resulting program with ``--vsip-profile-mode=accum`` or
/// ``--vsip-profile-mode=trace`` to see the profile output.

// This definition turns on recording of user-created profile events.
// It must be defined prior to including any Sourcery VSIPL++ headers.
#define VSIP_PROFILE_USER 1

#include <vsip/initfin.hpp>
#include <vsip_csl/profile.hpp>
#include <iostream>


// ====================================================================
// Main Program
int
main(int argc, char **argv)
{
  // Print a helpful message if the user didn't call this with any
  // profiling options.
  if (argc == 1)
  {
    std::cout << "Use --vsip-profile-mode=accum or --vsip-profile-mode=trace"
	      << std::endl;
    std::cout << "command-line options to see profiling output." << std::endl;
  }

  // Initialize the Sourcery VSIPL++ library.
  vsip::vsipl init(argc, argv);

  // ==================================================================
  // "Scope" profile objects measure the time between their
  // construction and their destruction at the end of the containing
  // scope.  Typically, this will be the scope of a function, but it
  // can also be simply a bare scope, as here.
  {
    // We define an (optional) operation count for the scope, which
    // will be used in computing ops/second numbers.  Normally this
    // would be arithmetic operations, but here we'll use milliseconds.
    int const op_count = 2000;

    // Create the Scope object.  Its timer starts on creation.
    vsip_csl::profile::Scope<vsip_csl::profile::user>
      scope("Scope 1", op_count);
    
    // Wait some time so as to get interesting profile numbers.
    sleep(2);
    
    // The Scope object is destroyed when it goes out of scope; this
    // stops its timer and delineates the code that it is timing.
  }

  // ==================================================================
  // "Event" profile function calls simply place an "event" record
  // in the trace profile data; they do not measure times.
  vsip_csl::profile::event<vsip_csl::profile::user>("An event");

  // ==================================================================
  // We'll include another scope with a Scope object measuring it, to
  // make the profile output more interesting.
  {
    vsip_csl::profile::Scope<vsip_csl::profile::user>
      scope("Scope 2", 1000);
    sleep(1);

    // Scope objects can be nested, and this nesting will show up in
    // the profile output.
    {
      vsip_csl::profile::Scope<vsip_csl::profile::user>
	scope("Nested Scope", 1000);
      sleep(1);
    }
  }
}
