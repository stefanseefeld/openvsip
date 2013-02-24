/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    support.cpp
    @author  Zack Weinberg
    @date    2005-01-26
    @brief   VSIPL++ Library: [support] basic macros, types, exceptions,
             and support functions (implementation).

    This file defines out-of-line functions and data objects declared in
    support.hpp.
*/

#include <vsip/support.hpp>
#include <iostream>
#include <cstdlib>

/// This function is called instead of throwing an exception when
/// VSIP_HAS_EXCEPTIONS is 0.

#if !VSIP_HAS_EXCEPTIONS
void
vsip::impl::fatal_exception(char const * file, unsigned int line,
                            std::exception const& E)
{
  std::cerr << "VSIPL++: at " << file << ':' << line << '\n';
  std::cerr << "VSIPL++: fatal: " << E.what() << std::endl;
  std::abort ();
}
#endif
