/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/

#ifndef vsip_csl_cbe_hpp_
#define vsip_csl_cbe_hpp_

#include <vsip/opt/cbe/ppu/task_manager.hpp>

namespace vsip_csl
{
namespace cbe
{
  inline void
  set_num_spes(int num_spes, bool verify = false)
  {
    vsip::impl::cbe::Task_manager::set_num_spes(num_spes, verify);
  }

  inline int
  get_num_spes()
  {
    if (vsip::impl::cbe::Task_manager::instance())
      return vsip::impl::cbe::Task_manager::instance()->num_spes();
    else
      return -1;
  }
}
}

#endif
