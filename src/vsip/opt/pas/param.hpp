/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/pas/param.hpp
    @author  Jules Bergmann
    @date    2006-08-09
    @brief   VSIPL++ Library: Parallel Services: PAS parameters

*/

#ifndef VSIP_PAS_PAS_PARAM_HPP
#define VSIP_PAS_PAS_PARAM_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>



/***********************************************************************
  Macros
***********************************************************************/

// Set VSIP_IMPL_PAS_XR to 1 when using PAS for Linux
#if PAS_RDMA
#  define VSIP_IMPL_PAS_XR                        1
#else
#  define VSIP_IMPL_PAS_XR                        0
#endif

#if VSIP_IMPL_PAS_XR
#  define VSIP_IMPL_PAS_XR_SET_PORTNUM            0
#  define VSIP_IMPL_PAS_XR_SET_ADAPTERNAME        1
#  define VSIP_IMPL_PAS_XR_SET_SHMKEY             1
#  define VSIP_IMPL_PAS_XR_SET_PIR                0
#  define VSIP_IMPL_PAS_XR_SET_RMD                0
#  define VSIP_IMPL_PAS_XR_SET_UDAPL_MAX_RECVS    0
#  define VSIP_IMPL_PAS_XR_SET_UDAPL_MAX_REQUESTS 0

#  define VSIP_IMPL_PAS_USE_INTERRUPT() 1
#  define VSIP_IMPL_PAS_XFER_ENGINE PAS_DMA
#  define VSIP_IMPL_PAS_XR_ADAPTERNAME "ib0" /* Commonly used with Mercury XR9 */
#  define VSIP_IMPL_PAS_XR_SHMKEY 1918
#else
#  define VSIP_IMPL_PAS_USE_INTERRUPT() 0
#  define VSIP_IMPL_PAS_XFER_ENGINE PAS_DMA
#endif

#define VSIP_IMPL_PAS_ALIGNMENT 16

#ifndef VSIP_IMPL_PAS_HEAP_SIZE
#  define VSIP_IMPL_PAS_HEAP_SIZE 0x100000
#endif

#if VSIP_IMPL_PAS_USE_INTERRUPT()
#  define VSIP_IMPL_PAS_SEM_GIVE_AFTER PAS_SEM_GIVE_INTERRUPT_AFTER
#else
#  define VSIP_IMPL_PAS_SEM_GIVE_AFTER PAS_SEM_GIVE_AFTER
#endif

#define VSIP_IMPL_CHECK_RC(rc, where)					\
  if (rc != CE_SUCCESS)							\
  {									\
     err_print(rc, ERR_GET_ALL);					\
     printf("CE%ld %s %s L%d\n", ce_getid(), where, __FILE__, __LINE__);\
     assert(0);								\
     abort();								\
  }

#endif // VSIP_PAS_PAS_PARAM_HPP
