/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/check_config.cpp
    @author  Jules Bergmann
    @date    2006-10-04
    @brief   VSIPL++ Library: Check library configuration.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <string>
#include <sstream>

#include <vsip/core/config.hpp>
#include <vsip/core/check_config.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

std::string
library_config()
{
  std::ostringstream   cfg;

#include <vsip/core/check_config_body.hpp>

  return cfg.str();
}

} // namespace vsip::impl
} // namespace vsip
