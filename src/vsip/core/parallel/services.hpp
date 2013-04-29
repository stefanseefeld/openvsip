//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PARALLEL_HPP
#define VSIP_CORE_PARALLEL_HPP

#include <vsip/core/config.hpp>

#if VSIP_IMPL_PAR_SERVICE == 1
#  include <vsip/core/mpi/services.hpp>
#else
#  include <vsip/core/parallel/services_none.hpp>
#endif

namespace vsip
{
namespace impl
{

/// Return the default communicator.
inline Communicator &default_communicator()
{
  return Par_service::default_communicator();
}

} // namespace vsip::impl

/// Return the number of processors in the data parallel clique.
inline length_type num_processors() VSIP_NOTHROW
{
  return impl::default_communicator().size();
}

/// Return the local processor.
inline processor_type local_processor() VSIP_NOTHROW
{
  return impl::default_communicator().rank();
}

} // namespace vsip

#endif // VSIP_CORE_PARALLEL_HPP
