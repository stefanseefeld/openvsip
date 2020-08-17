//
// Copyright (c) 2009 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cxx11_hpp_
#define ovxx_cxx11_hpp_

#include <ovxx/config.hpp>

namespace ovxx
{
using std::integral_constant;
using std::true_type;
using std::false_type;
using std::enable_if;
using std::conditional;
using std::is_same;
using std::is_const;
using std::is_volatile;
using std::remove_const;
using std::remove_volatile;
using std::remove_cv;
using std::add_const;
using std::add_volatile;
using std::add_cv;
using std::is_integral;
using std::is_floating_point;
using std::is_unsigned;
using std::is_arithmetic;
using std::is_signed;
using std::is_unsigned;
} // namespace ovxx

#endif
