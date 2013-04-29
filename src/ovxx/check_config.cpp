//
// Copyright (c) 2006, 2007, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/check_config.hpp>
#include <string>
#include <sstream>

namespace ovxx
{
std::string library_config()
{
  std::ostringstream   cfg;

#include <ovxx/detail/check_config.hpp>

  return cfg.str();
}

} // namespace ovxx
