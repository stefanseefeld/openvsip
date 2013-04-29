/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/bindings.hpp
    @author  Stefan Seefeld
    @date    2005-08-10
    @brief   VSIPL++ Library: Wrappers and traits to bridge with Intel's IPP.
*/

#ifndef VSIP_IMPL_IPP_HPP
#define VSIP_IMPL_IPP_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/dda.hpp>

namespace vsip
{
namespace impl
{
namespace ipp
{

template <typename Type>
struct is_type_supported
{
  static bool const value = false;
};

template <>
struct is_type_supported<float>
{
  static bool const value = true;
};

template <>
struct is_type_supported<double>
{
  static bool const value = true;
};

template <>
struct is_type_supported<std::complex<float> >
{
  static bool const value = true;
};

template <>
struct is_type_supported<std::complex<double> >
{
  static bool const value = true;
};

// functions for vector copy
void vcopy(float const* A, float* Z, length_type len);
void vcopy(double const* A, double* Z, length_type len);
void vcopy(complex<float> const* A, complex<float>* Z, length_type len);
void vcopy(complex<double> const* A, complex<double>* Z, length_type len);

// Vector zero
void vzero(float*           Z, length_type len);
void vzero(double*          Z, length_type len);
void vzero(complex<float>*  Z, length_type len);
void vzero(complex<double>* Z, length_type len);

// functions for convolution
void conv(float const *coeff, length_type coeff_size,
	  float const *in, length_type in_size,
	  float* out);
void conv(double const *coeff, length_type coeff_size,
	  double const *in, length_type in_size,
	  double* out);

void
conv_full_2d(float const *coeff,
	     length_type coeff_rows,
	     length_type coeff_cols,
	     length_type coeff_row_stride,
	     float const *in,
	     length_type in_rows,
	     length_type in_cols,
	     length_type in_row_stride,
	     float*      out,
	     length_type out_row_stride);

void
conv_full_2d(short const *coeff,
	     length_type coeff_rows,
	     length_type coeff_cols,
	     length_type coeff_row_stride,
	     short const *in,
	     length_type in_rows,
	     length_type in_cols,
	     length_type in_row_stride,
	     short*      out,
	     length_type out_row_stride);

void
conv_valid_2d(float const *coeff,
	      length_type coeff_rows,
	      length_type coeff_cols,
	      length_type coeff_row_stride,
	      float const *in,
	      length_type in_rows,
	      length_type in_cols,
	      length_type in_row_stride,
	      float*      out,
	      length_type out_row_stride);

void
conv_valid_2d(short const *coeff,
	      length_type coeff_rows,
	      length_type coeff_cols,
	      length_type coeff_row_stride,
	      short const *in,
	      length_type in_rows,
	      length_type in_cols,
	      length_type in_row_stride,
	      short*      out,
	      length_type out_row_stride);

} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip

#endif
