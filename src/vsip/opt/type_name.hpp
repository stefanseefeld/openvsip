/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/type_name.hpp
    @author  Stefan Seefeld
    @date    2009-07-15
    @brief   VSIPL++ Library: Demangle typenames if possible.

*/

#ifndef VSIP_OPT_TYPE_NAME_HPP
#define VSIP_OPT_TYPE_NAME_HPP

#include <string>
#include <typeinfo>

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

namespace vsip
{
namespace impl
{
std::string demangle(char const *mangled);

/// Return the type-name of type 'T'
template <typename T>
std::string type_name()
{
  return demangle(typeid(T).name());
}

/// Return the type-name of argument 't'
template <typename T>
std::string type_name(T const &t)
{
  return demangle(typeid(t).name());
}

} // namespace vsip::impl
} // namespace vsip

#endif
