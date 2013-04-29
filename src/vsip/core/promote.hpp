/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/promote.hpp
    @author  Stefan Seefeld
    @date    2005-01-25
    @brief   VSIPL++ Library: [math.fns.promotion] Promotion traits.

    This file defines Promotion traits used to declare value types for
    arithmetic expressions.
*/

#ifndef VSIP_CORE_PROMOTE_HPP
#define VSIP_CORE_PROMOTE_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <cmath>
#include <complex>

namespace vsip
{

/***********************************************************************
  Declarations
***********************************************************************/

/// [math.fns.promotion]
template <typename Left, typename Right>
struct Promotion
{
  typedef Left type;
};

template <typename Left, typename Right>
struct Promotion<std::complex<Left>, std::complex<Right> >
{
  typedef std::complex<typename Promotion<Left, Right>::type> type;
};

template <typename Left, typename Right>
struct Promotion<std::complex<Left>, Right>
{
  typedef std::complex<typename Promotion<Left, Right>::type> type;
};

template <typename Left, typename Right>
struct Promotion<Left, std::complex<Right> >
{
  typedef std::complex<typename Promotion<Left, Right>::type> type;
};

template <> struct Promotion<bool, char> { typedef char type;};
template <> struct Promotion<bool, short> { typedef short type;};
template <> struct Promotion<bool, int> { typedef int type;};
template <> struct Promotion<bool, long> { typedef long type;};
template <> struct Promotion<bool, float> { typedef float type;};
template <> struct Promotion<bool, double> { typedef double type;};

template <> struct Promotion<short, int> { typedef int type;};
template <> struct Promotion<short, long> { typedef long type;};
template <> struct Promotion<short, float> { typedef float type;};
template <> struct Promotion<short, double> { typedef double type;};

template <> struct Promotion<int, long> { typedef long type;};
template <> struct Promotion<int, float> { typedef float type;};
template <> struct Promotion<int, double> { typedef double type;};

template <> struct Promotion<long, float> { typedef float type;};
template <> struct Promotion<long, double> { typedef double type;};

template <> struct Promotion<float, double> { typedef double type;};

}

#endif
