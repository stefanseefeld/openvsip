/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ukernel/cbe_accel/debug.hpp
    @author  Don McCoy
    @date    2008-08-27
    @brief   VSIPL++ Library: User-defined Kernel, debug routines.
*/

#ifndef VSIP_OPT_UKERNEL_CBE_ACCEL_DEBUG_HPP
#define VSIP_OPT_UKERNEL_CBE_ACCEL_DEBUG_HPP

#include <vsip/opt/ukernel/cbe_accel/ukernel.hpp>

#ifndef NDEBUG
#include <stdlib.h>

void cbe_debug_dump_pinfo(char tag[], Pinfo const& pinfo) 
{
  printf("Pinfo %s = \n", 
    tag);
  printf(" dim:           %u\n", 
    pinfo.dim );
  printf(" l_total_size:  %u\n", 
    pinfo.l_total_size );
  printf(" l_offset[3]:   %u  %u  %u\n",
    pinfo.l_offset[0], 
    pinfo.l_offset[1], 
    pinfo.l_offset[2]);
  printf(" l_size[3]:     %u  %u  %u\n", 
    pinfo.l_size[0], 
    pinfo.l_size[1], 
    pinfo.l_size[2]);
  printf(" l_stride[3]:   %d  %d  %d\n", 
    pinfo.l_stride[0], 
    pinfo.l_stride[1], 
    pinfo.l_stride[2]);
  printf(" g_offset[3]:   %d  %d  %d\n", 
    pinfo.g_offset[0], 
    pinfo.g_offset[1], 
    pinfo.g_offset[2]);
  printf(" o_leading[3]:  %d  %d  %d\n", 
    pinfo.o_leading[0], 
    pinfo.o_leading[1], 
    pinfo.o_leading[2]);
  printf(" o_trailing[3]: %d  %d  %d\n", 
    pinfo.o_trailing[0], 
    pinfo.o_trailing[1], 
    pinfo.o_trailing[2]);
}


#else

inline void cbe_debug_dump_pinfo(Pinfo const&) {}


#endif // NDEBUG

#endif // VSIP_OPT_UKERNEL_CBE_ACCEL_DEBUG_HPP
