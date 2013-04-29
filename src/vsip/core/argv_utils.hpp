//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_ARGV_UTILS_HPP
#define VSIP_CORE_ARGV_UTILS_HPP

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

inline void
shift_argv(int& argc, char**&argv, int pos, int shift)
{
  for (int i=pos; i<argc-shift; ++i)
    argv[i] = argv[i+shift];
  argc -= shift;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_ARGV_UTILS_HPP
