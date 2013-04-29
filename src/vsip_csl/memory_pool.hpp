/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/memory_pool.hpp
    @author  Jules Bergmann
    @date    2008-11-21
    @brief   VSIPL++ Library: CSL extension: Memory pools
*/

#ifndef VSIP_CSL_MEMORY_POOL_HPP
#define VSIP_CSL_MEMORY_POOL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/memory_pool.hpp>
#include <vsip/core/huge_page_pool.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{

using vsip::impl::Memory_pool;
using vsip::impl::Huge_page_pool;

void
set_pool(vsip::Local_map& map, Memory_pool* pool)
{
  map.impl_set_pool(pool);
}

} // namespace vsip_csl

#endif // VSIP_CSL_UKERNEL_HPP
