//
// Copyright (c) 2005 - 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_config_hpp_
#define ovxx_config_hpp_

#ifdef OVXX_VARIANT
# define OVXX_ACCONFIG <ovxx/detail/config-OVXX_VARIANT.hpp>
# include OVXX_ACCONFIG
#else
# include <ovxx/detail/config.hpp>
#endif

#ifdef  _MSC_VER

// Make sure we see the POSIX definitions from math.h.
# define _USE_MATH_DEFINES

#endif

#ifdef M_PI
# define OVXX_PI M_PI
#else
# define OVXX_PI 3.14159265358979323846
#endif

#if defined(_WIN32)
// IPP on Windows uses __stdcall for all functions.
# define OVXX_IPP_CALL __stdcall
#else
# define OVXX_IPP_CALL
#endif

// Remove macros that autoconf sometimes defines.
#ifdef SIZEOF_DOUBLE
#  undef SIZEOF_DOUBLE
#endif

#ifdef SIZEOF_LONG_DOUBLE
#  undef SIZEOF_LONG_DOUBLE
#endif

#ifdef SIZEOF_LONG
#  undef SIZEOF_LONG
#endif

#ifdef __GNUC__
#define OVXX_UNUSED __attribute__((unused))
#else
#define OVXX_UNUSED /* empty */
#endif

#ifdef OVXX_HAVE_MPI
# define OVXX_PARALLEL 1
#endif

#ifndef OVXX_TUNE_MODE
/// Setting TUNE_MODE to 1 disables the tunable_thresholds.  This
/// allows benchmarks to be run without thresholds to determine
/// cross-over points, which can then be used as thresholds.
#  define OVXX_TUNE_MODE 0
#endif

#endif
