//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_library_hpp_
#define ovxx_library_hpp_

#include <ovxx/detail/noncopyable.hpp>

namespace ovxx
{

class library : detail::noncopyable
{
public:
  library();
  library(int& argc, char**& argv);
  ~library();
};

} // namespace ovxx

#endif
