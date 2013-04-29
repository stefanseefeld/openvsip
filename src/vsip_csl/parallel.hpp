/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_CSL_PARALLEL_HPP
#define VSIP_CSL_PARALLEL_HPP

#include <vsip/core/parallel/services.hpp>

#if VSIP_IMPL_PAR_SERVICE != 1
# error "The parallel API is not supported by the current configuration."
#endif

namespace vsip_csl
{
namespace parallel
{
  
using vsip::impl::mpi::Communicator;
using vsip::impl::mpi::Group;
using vsip::impl::default_communicator;
using vsip::impl::mpi::set_default_communicator;

}
}

#endif
