/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/pas/util.hpp
    @author  Jules Bergmann
    @date    2006-08-29
    @brief   VSIPL++ Library: Parallel Services: PAS utilities
*/

#ifndef VSIP_OPT_PAS_UTIL_HPP
#define VSIP_OPT_PAS_UTIL_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/opt/pas/param.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace pas
{

inline void
semaphore_give(PAS_id pset, long index)
{
  long rc;
#if VSIP_IMPL_PAS_USE_INTERRUPT()
  rc = pas_semaphore_give_interrupt(pset, index, VSIP_IMPL_PAS_XFER_ENGINE); 
#else
  rc = pas_semaphore_give(pset, index, VSIP_IMPL_PAS_XFER_ENGINE); 
#endif
  assert(rc == CE_SUCCESS);
}



inline void
semaphore_take(PAS_id pset, long index)
{
  long rc;
#if VSIP_IMPL_PAS_USE_INTERRUPT()
  rc = pas_semaphore_take_interrupt(pset, index, VSIP_IMPL_PAS_XFER_ENGINE); 
#else
  rc = pas_semaphore_take(pset, index, VSIP_IMPL_PAS_XFER_ENGINE); 
#endif
  assert(rc == CE_SUCCESS);
}

} // namespace vsip::impl::pas
} // namespace vsip::impl
} // namespace vsip
#endif // VSIP_OPT_PAS_UTIL_HPP
