//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_detail_endian_hpp_
#define ovxx_detail_endian_hpp_

#if defined (__GLIBC__)
# include <endian.h>
# if (__BYTE_ORDER == __LITTLE_ENDIAN)
#  define OVXX_LITTLE_ENDIAN
# elif (__BYTE_ORDER == __BIG_ENDIAN)
#  define OVXX_BIG_ENDIAN
# elif (__BYTE_ORDER == __PDP_ENDIAN)
#  define OVXX_PDP_ENDIAN
# else
#  error Unknown machine endianness detected.
# endif
# define OVXX_BYTE_ORDER __BYTE_ORDER

#elif defined(__NetBSD__) || defined(__FreeBSD__) || \
    defined(__OpenBSD__) || (__DragonFly__)
# if defined(__OpenBSD__)
#  include <machine/endian.h>
# else
#  include <sys/endian.h>
# endif
# if (_BYTE_ORDER == _LITTLE_ENDIAN)
#  define OVXX_LITTLE_ENDIAN
# elif (_BYTE_ORDER == _BIG_ENDIAN)
#  define OVXX_BIG_ENDIAN
# elif (_BYTE_ORDER == _PDP_ENDIAN)
#  define OVXX_PDP_ENDIAN
# else
#  error Unknown machine endianness detected.
# endif
# define OVXX_BYTE_ORDER _BYTE_ORDER

#elif defined(__sparc) || defined(__sparc__) \
   || defined(_POWER) || defined(__powerpc__) \
   || defined(__ppc__) || defined(__hpux) || defined(__hppa) \
   || defined(_MIPSEB) || defined(_POWER) \
   || defined(__s390__) || defined(__ARMEB__)
# define OVXX_BIG_ENDIAN
# define OVXX_BYTE_ORDER 4321
#elif defined(__i386__) || defined(__alpha__) \
   || defined(__ia64) || defined(__ia64__) \
   || defined(_M_IX86) || defined(_M_IA64) \
   || defined(_M_ALPHA) || defined(__amd64) \
   || defined(__amd64__) || defined(_M_AMD64) \
   || defined(__x86_64) || defined(__x86_64__) \
   || defined(_M_X64) || defined(__bfin__) \
   || defined(__ARMEL__) \
   || (defined(_WIN32) && defined(__ARM__) && defined(_MSC_VER))
# define OVXX_LITTLE_ENDIAN
# define OVXX_BYTE_ORDER 1234
#else
# error The file ovxx/endian.hpp needs to be set up for your CPU type.
#endif

namespace ovxx
{
bool const is_big_endian =
#if OVXX_BIG_ENDIAN
  true
#else
  false
#endif
  ;
}

#endif
