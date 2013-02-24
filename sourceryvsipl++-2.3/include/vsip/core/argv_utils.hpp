/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.
*/
/** @file    vsip/core/argv_utils.hpp
    @author  Jules Bergmann
    @date    2007-02-24
    @brief   VSIPL++ Library: Utils for mucking with argv.

*/

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
