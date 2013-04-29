/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    vsip_csl/cvsip/type_conversion.hpp
    @author  Stefan Seefeld
    @date    2008-06-03
    @brief   Define value-type conversion
*/
#ifndef vsip_csl_cvsip_type_conversion_hpp_
#define vsip_csl_cvsip_type_conversion_hpp_

#include <vsip.h>

namespace vsip_csl
{
namespace cvsip
{

template <typename PublicT, typename ImplT>
struct Converter
{
  typedef PublicT public_type;
  typedef ImplT impl_type;

  static impl_type *
  in(public_type *d) { return reinterpret_cast<impl_type*>(d);}
  static impl_type
  in(public_type d) { return d;}
  static public_type *
  out(impl_type *d) { return reinterpret_cast<public_type*>(d);}
  static public_type
  out(impl_type d) { return d;}
};

template <typename B> struct type_conversion;
template <> 
struct type_conversion<vsip_block_bl> : Converter<vsip_scalar_bl, int> {};
template <> 
struct type_conversion<vsip_block_i> : Converter<vsip_scalar_i, int> {};
template <> 
struct type_conversion<vsip_block_si> : Converter<vsip_scalar_si, signed short int> {};
template <> 
struct type_conversion<vsip_block_uc> : Converter<vsip_scalar_uc, unsigned char> {};
template <> 
struct type_conversion<vsip_block_f> : Converter<vsip_scalar_f, float> {};
template <> 
struct type_conversion<vsip_block_d> : Converter<vsip_scalar_d, double> {};
template <> 
struct type_conversion<vsip_cblock_f> : Converter<vsip_cscalar_f, std::complex<float> >
{
  using Converter<vsip_cscalar_f, std::complex<float> >::in;
  static impl_type
  in(public_type d) { return impl_type(d.r, d.i);}
  using Converter<vsip_cscalar_f, std::complex<float> >::out;
  static public_type
  out(impl_type d) { return vsip_cmplx_f(d.real(), d.imag());}
};
template <> 
struct type_conversion<vsip_cblock_d> : Converter<vsip_cscalar_d, std::complex<double> >
{
  using Converter<vsip_cscalar_d, std::complex<double> >::in;
  static impl_type
  in(public_type d) { return impl_type(d.r, d.i);}
  using Converter<vsip_cscalar_d, std::complex<double> >::out;
  static public_type
  out(impl_type d) { return vsip_cmplx_d(d.real(), d.imag());}
};
template <> 
struct type_conversion<vsip_block_vi> : Converter<vsip_scalar_vi, vsip::Index<1> > 
{
  using Converter<vsip_scalar_vi, vsip::Index<1> >::out;
  static public_type
  out(impl_type d) { return d[0];}
};
template <> 
struct type_conversion<vsip_block_mi> : Converter<vsip_scalar_mi, vsip::Index<2> >
{
  using Converter<vsip_scalar_mi, vsip::Index<2> >::in;
  static impl_type
  in(public_type d) { return impl_type(d.r, d.c);}
  using Converter<vsip_scalar_mi, vsip::Index<2> >::out;
  static public_type
  out(impl_type d) { return vsip_matindex(d[0], d[1]);}
};
template <> 
struct type_conversion<vsip_block_ti> : Converter<vsip_scalar_ti, vsip::Index<3> >
{
  using Converter<vsip_scalar_ti, vsip::Index<3> >::in;
  static impl_type
  in(public_type d) { return impl_type(d.z, d.y, d.x);}
  using Converter<vsip_scalar_ti, vsip::Index<3> >::out;
  static public_type
  out(impl_type d) { return vsip_tenindex(d[0], d[1], d[2]);}
};

} // namespace vsip_csl::cvsip
} // namespace vsip_csl

#endif
