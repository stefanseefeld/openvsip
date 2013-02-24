/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/type_name.cpp
    @author  Stefan Seefeld
    @date    2009-07-15
    @brief   VSIPL++ Library: Demangle typenames if possible.

*/
#include <vsip/opt/type_name.hpp>
#include <memory>
#include <cstdlib>
#include <cassert>

#include <memory>
#if defined(__GNUC__) &&  __GNUC__ >= 3
# include <cxxabi.h>

# if __GNUC__ == 3 && __GNUC_MINOR__ == 0
namespace abi
{
  extern "C" char* __cxa_demangle(char const*, char*, std::size_t*, int*);
}
# endif 
#endif


namespace vsip
{
namespace impl
{
namespace
{

struct free_mem
{
  free_mem(char *p) : p(p) {}
  ~free_mem() { std::free(p);}
  char * p;
};
} // anonymous namespace

std::string demangle(char const *mangled)
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

} // namespace vsip::impl
} // namespace vsip

