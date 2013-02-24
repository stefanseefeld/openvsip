/* Copyright (c)  2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/assign.hpp
    @author  Jules Bergmann
    @date    2006-07-14
    @brief   VSIPL++ Library: Parallel assignment class.

*/

#ifndef VSIP_CORE_PARALLEL_ASSIGN_HPP
#define VSIP_CORE_PARALLEL_ASSIGN_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/config.hpp>
#include <vsip/core/parallel/assign_fwd.hpp>
#include <vsip/core/parallel/choose_assign_impl.hpp>

#if VSIP_IMPL_PAR_SERVICE == 0 || VSIP_IMPL_PAR_SERVICE == 1
#  include <vsip/core/parallel/assign_chain.hpp>
#  include <vsip/core/parallel/assign_block_vector.hpp>
#elif VSIP_IMPL_PAR_SERVICE == 2
#  include <vsip/opt/pas/assign.hpp>
#  include <vsip/opt/pas/assign_eb.hpp>
#  include <vsip/opt/pas/assign_direct.hpp>
#endif



#endif // VSIP_IMPL_PAR_ASSIGN_HPP
