/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/numa.cpp
    @author  Stefan Seefeld
    @date    2007-03-06
    @brief   VSIPL++ Library: 
*/
 
#include <vsip/core/argv_utils.hpp>
#include <vsip/opt/numa.hpp>
#include <numa.h>

namespace vsip
{
namespace impl
{
namespace numa
{
void local_spes_only()
{
  nodemask_t mask;
  nodemask_zero(&mask);
  nodemask_set(&mask, 1);
  numa_bind(&mask);
}

void initialize(int argc, char **&argv)
{
  int count = argc;
  char** value = argv;
  while(--count)
  {
    ++value;
    if (!strcmp(*value, "--vsip-local-spes-only"))
    {
      shift_argv(argc, argv, argc - count, 1);
      local_spes_only();
    }
  }
}
}
}
}
