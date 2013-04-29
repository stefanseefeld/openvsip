//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

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
