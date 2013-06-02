//
// Copyright (c) 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_inttypes_hpp_
#define ovxx_inttypes_hpp_

#include <ovxx/config.hpp>

#if HAVE_STDINT_H
# include <stdint.h>
#endif

namespace ovxx
{
#if HAVE_STDINT_H
  typedef int8_t int8_type;
  typedef uint8_t uint8_type;
  typedef int16_t int16_type;
  typedef uint16_t uint16_type;
  typedef int32_t int32_type;
  typedef uint32_t uint32_type;
  typedef int64_t int64_type;
  typedef uint64_t uint64_type;
#else

  typedef signed char int8_type;
  typedef unsigned char uint8_type;

# if SIZEOF_SHORT == 2
  typedef short int16_type;
  typedef unsigned short uint16_type;
# elif SIZEOF_INT == 2
  typedef int int16_type;
  typedef unsigned int uint16_type;
# else
#  error "No 16-bit integer type"
# endif

# if SIZEOF_SHORT == 4
  typedef unsigned short uint32_type;
  typedef short int32_type;
# elif SIZEOF_INT == 4
  typedef unsigned int uint32_type;
  typedef int int32_type;
# elif SIZEOF_LONG == 4
  typedef unsigned long uint32_type;
  typedef long int32_type;
# else
#  error "No 32-bit integer type"
# endif

# if SIZEOF_INT == 8
  typedef unsigned int uint64_type;
  typedef int int64_type;
# elif SIZEOF_LONG == 8
  typedef unsigned long uint64_type;
  typedef long int64_type;
# elif SIZEOF_LONG_LONG == 8
  typedef unsigned long long uint64_type;
  typedef long long int64_type;
# else
#  error "No 64-bit integer type"
# endif
#endif 
} // namespace ovxx

#endif
