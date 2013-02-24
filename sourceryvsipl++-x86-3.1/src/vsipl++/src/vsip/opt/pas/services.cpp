/* Copyright (c) 2010 CodeSourcery.  All rights reserved.  */

#include "services.hpp"

namespace vsip
{
namespace impl
{
namespace pas
{

/// Counter to generate a unique tag for global PAS pbuffer allocations.
long             global_tag = 1;

} // namespace vspi::impl::pas

Par_service::communicator_type Par_service::default_communicator_;

} // namespace vsip::impl
} // namespace vsip
