//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
