/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/check_config.hpp
    @author  Jules Bergmann
    @date    2006-10-04
    @brief   VSIPL++ Library: Check library configuration.
*/

#ifndef VSIP_IMPL_CHECK_CONFIG_HPP
#define VSIP_IMPL_CHECK_CONFIG_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <string>
#include <sstream>

#include <vsip/core/config.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

// Configuration when library was built.
std::string library_config();

// Configuration when application was built.
inline std::string
app_config()
{
  std::ostringstream   cfg;

#include <vsip/core/check_config_body.hpp>

  return cfg.str();
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_CHECK_CONFIG_HPP
