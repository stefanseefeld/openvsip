/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef CSIP_CSL_TEST_DISPATCH_HPP
#define CSIP_CSL_TEST_DISPATCH_HPP

#include <vsip/opt/dispatch.hpp>
#include <iostream>

namespace vsip_csl
{

void clear_dispatch_trace()
{
  vsip::impl::profile::prof->clear();
}

/// Print a trace of scopes entered during dispatch.
void print_dispatch_trace()
{
  using vsip::impl::profile::Profiler;
  using vsip::impl::profile::prof;
  std::cout << "dispatch trace:" << std::endl;
  for (Profiler::trace_type::iterator i = prof->begin_trace();
       i != prof->end_trace();
       ++i)
    if (i->end != 0) // only print enter events.
      std::cout << "'" << i->name << "'" << std::endl;
}

// The reference in the function argument is required
// to prevent the array type from decaying into a pointer.
// [14.8.2.1] para 2

/// Compare the dispatch trace with a list of expected
/// scopes. Useful to validate that an expected backend
/// has been used in a dispatch call.
template <unsigned int N>
void validate_dispatch_trace(char const *(&entries)[N])
{
  using vsip::impl::profile::Profiler;
  using vsip::impl::profile::prof;
  Profiler::trace_type::iterator i = prof->begin_trace();
  bool valid = true;
  // Compare expected entries to actual entries
  for (unsigned int j = 0; valid && j != N; ++j, ++i)
    valid = i != prof->end_trace() && i->name == entries[j];
  // The remainder of the trace should only contain stack leave events.
  while (valid && i != prof->end_trace())
    valid = (i++)->end != 0; // '0' values indicates enter event.

  if (!valid)
  {
    std::cerr << "dispatch trace validation failure!\n"
              << "  expected:\n";
    for (unsigned int j = 0; j != N; ++j) std::cerr << "    " << entries[j] << '\n';
    std::cerr << "  got:\n";
    i = prof->begin_trace();
    for (unsigned int j = 0; j != N && i != prof->end_trace(); ++j, ++i)
      std::cerr << "    " << i->name << '\n';
    while (valid && i != prof->end_trace())
      if ((i++)->end == 0)
        std::cerr << "    " << i->name << '\n';
    std::cerr.flush();
  }
  test_assert(valid);
  prof->clear();
}
}

#endif
