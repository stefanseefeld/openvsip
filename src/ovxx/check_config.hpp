//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_check_config_hpp_
#define ovxx_check_config_hpp_

#include <ovxx/config.hpp>
#include <string>
#include <sstream>

namespace ovxx
{

/// Report configuration when library was built.
std::string library_config();

/// Report configuration when application was built.
inline std::string app_config()
{
  std::ostringstream cfg;

#include <ovxx/detail/check_config.hpp>

  return cfg.str();
}

} // namespace ovxx

#endif
