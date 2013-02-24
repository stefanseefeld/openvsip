/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/numa.hpp
    @author  Stefan Seefeld
    @date    2007-03-06
    @brief   VSIPL++ Library: Provide access to the libnuma API
*/

#ifndef VSIP_OPT_NUMA_HPP
#define VSIP_OPT_NUMA_HPP

namespace vsip
{
namespace impl
{
namespace numa
{
void initialize(int argv, char **&argv);
}
}
}

#endif
