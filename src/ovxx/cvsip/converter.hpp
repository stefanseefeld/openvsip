//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_converter_hpp_
#define ovxx_cvsip_converter_hpp_

extern "C" {
#include <vsip.h>
}

namespace ovxx
{
namespace cvsip
{

template <typename T>
struct converter
{
  typedef T vsiplxx_type;
  typedef T vsipl_type;

  static vsiplxx_type to_vsiplxx(vsipl_type value)
  { return static_cast<vsiplxx_type>(value);}

  static vsipl_type to_vsipl(vsiplxx_type value)
  { return static_cast<vsipl_type>(value);}
};

template <>
struct converter<complex<float> >
{
  typedef complex<float> vsiplxx_type;
  typedef vsip_cscalar_f vsipl_type;

  static vsiplxx_type to_vsiplxx(vsipl_type value)
  { return vsiplxx_type(value.r, value.i);}

  static vsipl_type to_vsipl(vsiplxx_type value)
  {
    vsipl_type v = { value.real(), value.imag()};
    return v;
  }
};

template <>
struct converter<complex<double> >
{
  typedef complex<double> vsiplxx_type;
  typedef vsip_cscalar_d vsipl_type;

  static vsiplxx_type to_vsiplxx(vsipl_type value)
  { return vsiplxx_type(value.r, value.i);}

  static vsipl_type to_vsipl(vsiplxx_type value)
  {
    vsipl_type v = { value.real(), value.imag()};
    return v;
  }
};

template <>
struct converter<Index<1> >
{
  typedef Index<1> vsiplxx_type;
  typedef vsip_scalar_vi vsipl_type;

  static vsiplxx_type to_vsiplxx(vsipl_type value)
  { return vsiplxx_type(value);}

  static vsiplxx_type to_vsipl(vsiplxx_type value)
  {
    return vsipl_type(value[0]);
  }
};

template <>
struct converter<Index<2> >
{
  typedef Index<2> vsiplxx_type;
  typedef vsip_scalar_mi vsipl_type;

  static vsiplxx_type to_vsiplxx(vsipl_type value)
  { return vsiplxx_type(value.r, value.c);}

  static vsipl_type to_vsipl(vsiplxx_type value)
  {
    vsipl_type v = { value[0], value[1]};
    return v;
  }
};

} // namespace ovxx::cvsip
} // namespace ovxx

#endif
