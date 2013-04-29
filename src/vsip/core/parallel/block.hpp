/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/block.hpp
    @author  Jules Bergmann
    @date    2007-02-06
    @brief   VSIPL++ Library: Common header for parallel block.

*/

#ifndef VSIP_CORE_PARALLEL_BLOCK_HPP
#define VSIP_CORE_PARALLEL_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>

#if VSIP_IMPL_PAR_SERVICE == 1
#  include <vsip/core/parallel/distributed_block.hpp>
#elif VSIP_IMPL_PAR_SERVICE == 2
#  include <vsip/opt/pas/block.hpp>
#else
// If PAR_SERVICE == 0, Distributed_block is used by default.
#  include <vsip/core/parallel/distributed_block.hpp>
#endif

#endif // VSIP_CORE_PARALLEL_BLOCK_HPP
