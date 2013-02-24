/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/cvsip/convert_value.hpp
    @author  Jules Bergmann
    @date    2006-12-07
    @brief   VSIPL++ Library: Convert between C-VSIP and C++ value types.

*/

#ifndef VSIP_CORE_CVSIP_CONVERT_VALUE_HPP
#define VSIP_CORE_CVSIP_CONVERT_VALUE_HPP

/***********************************************************************
  Included Files
***********************************************************************/

extern "C" {
#include <vsip.h>
}




/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace cvsip
{

template <typename T>
struct Convert_value
{
  typedef T cpp_type;
  typedef T cvsip_type;

  static cpp_type to_cpp(cvsip_type value)
  { return static_cast<cpp_type>(value); }

  static cvsip_type to_cvsip(cpp_type value)
  { return static_cast<cvsip_type>(value); }
};

template <>
struct Convert_value<complex<float> >
{
  typedef complex<float> cpp_type;
  typedef vsip_cscalar_f cvsip_type;

  static cpp_type to_cpp(cvsip_type value)
  { return cpp_type(value.r, value.i); }

  static cvsip_type to_cvsip(cpp_type value)
  {
    cvsip_type v = { value.real(), value.imag() };
    return v;
  }
};

template <>
struct Convert_value<complex<double> >
{
  typedef complex<double> cpp_type;
  typedef vsip_cscalar_d cvsip_type;

  static cpp_type to_cpp(cvsip_type value)
  { return cpp_type(value.r, value.i); }

  static cvsip_type to_cvsip(cpp_type value)
  {
    cvsip_type v = { value.real(), value.imag() };
    return v;
  }
};



template <>
struct Convert_value<Index<1> >
{
  typedef Index<1> cpp_type;
  typedef vsip_scalar_vi cvsip_type;

  static cpp_type to_cpp(cvsip_type value)
  { return cpp_type(value); }

  static cvsip_type to_cvsip(cpp_type value)
  {
    return cvsip_type(value[0]);
  }
};



template <>
struct Convert_value<Index<2> >
{
  typedef Index<2> cpp_type;
  typedef vsip_scalar_mi cvsip_type;

  static cpp_type to_cpp(cvsip_type value)
  { return cpp_type(value.r, value.c); }

  static cvsip_type to_cvsip(cpp_type value)
  {
    cvsip_type v = { value[0], value[1] };
    return v;
  }
};

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

#endif
