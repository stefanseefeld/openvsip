/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description: VSIPL++ Library: Assignment diagnostics.
///
/// This example illustrates the use of diagnostics to see what
/// backend a given expression is evaluated with during assignment.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <iostream>

int
main(int argc, char **argv)
{
  // Initialize the Sourcery VSIPL++ library.
  vsip::vsipl init(argc, argv);

  vsip::Vector<> A(8);
  vsip::Vector<> B(8);
  vsip::Vector<> C(8);

  A = B * C;
  vsip_csl::assign_diagnostics(A, B * C);
}
