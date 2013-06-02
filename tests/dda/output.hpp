//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef dda_output_hpp_
#define dda_output_hpp_

#include <string>
#include <sstream>
#include <complex>

#include <vsip/support.hpp>
#include <vsip/dda.hpp>
#include <ovxx/type_name.hpp>

std::string access_type(ovxx::dda::access_kind k)
{
  using namespace ovxx::dda;
  switch (k)
  {
    case direct_access: return "direct";
    case copy_access: return "copy";
    case maybe_direct_access: return "maybe_direct";
    case local_access: return "local";
    case remote_access: return "remote";
    default: return "unknown";
  }
}
template <typename B, typename L>
std::string
access_type()
{
  return access_type(ovxx::dda::get_block_access<B, L, vsip::dda::in>::value);
}

template <typename L, typename B>
std::string
access_type(B const &) { return access_type<B, L>();}

template <typename LP>
void
print_layout(std::ostream& out)
{
  out << "  dim  = " << LP::dim << std::endl
      << "  pack  = " << LP::packing << std::endl
      << "  order = " << ovxx::type_name<typename LP::order_type>() << std::endl
      << "  storage = " << LP::storage_format
      << std::endl
    ;
}

#endif
