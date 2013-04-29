//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
