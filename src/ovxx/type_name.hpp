//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_type_name_hpp_
#define ovxx_type_name_hpp_

#include <string>
#include <typeinfo>
#include <memory>
#include <cstdlib>
#include <cassert>

#include <memory>
#if defined(__GNUC__) &&  __GNUC__ >= 3
# include <cxxabi.h>
#endif

namespace ovxx
{
namespace detail
{
struct free_mem
{
  free_mem(char *p) : p(p) {}
  ~free_mem() { std::free(p);}
  char * p;
};
inline std::string demangle(char const *mangled)
{
#if defined(__GNUC__) &&  __GNUC__ >= 3
  std::string demangled;
  int status;
  free_mem keeper(abi::__cxa_demangle(mangled, 0, 0, &status));
  assert(status != -3); // invalid argument error
  if (status == -1) { throw std::bad_alloc();}
  else
    // On failure return the mangled name.
    demangled = status == -2 ? mangled : keeper.p;
  return demangled;
#else
  return mangled;
#endif
}

} // namespace ovxx::detail

/// Return the type-name of type 'T'
template <typename T>
std::string type_name()
{
  return detail::demangle(typeid(T).name());
}

/// Return the type-name of argument 't'
template <typename T>
std::string type_name(T const &t)
{
  return detail::demangle(typeid(t).name());
}

} // namespace ovxx

#endif
