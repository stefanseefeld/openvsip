/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/bindings.hpp
    @author  Stefan Seefeld
    @date    2005-08-10
    @brief   VSIPL++ Library: Wrappers and traits to bridge with Intel's IPP.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip/opt/ipp/bindings.hpp>
#include <vsip/core/coverage.hpp>
#include <ipps.h>
#include <ippi.h>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace ipp
{


// Generate a vector from scratch
#define VSIP_IMPL_IPP_VGEN0(FCN, T, IPPFCN, IPPT)			\
void									\
FCN(									\
  T*       Z,								\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("IPP_VGEN0", IPPFCN)				\
  if (len > 0)								\
  {									\
    IppStatus status = IPPFCN(						\
      reinterpret_cast<IPPT*>(Z),					\
      static_cast<int>(len));						\
    assert(status == ippStsNoErr);					\
  }									\
}



#define VSIP_IMPL_IPP_V(FCN, T, IPPFCN, IPPT)				\
void									\
FCN(									\
  T const* A,								\
  T*       Z,								\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("IPP_V", IPPFCN)					\
  if (len > 0)								\
  {									\
    IppStatus status = IPPFCN(						\
      reinterpret_cast<IPPT const*>(A),					\
      reinterpret_cast<IPPT*>(Z),					\
      static_cast<int>(len));						\
    assert(status == ippStsNoErr);					\
  }									\
}

// Complex->Real unary function.

#define VSIP_IMPL_IPP_V_CR(FCN, T, IPPFCN, IPPCT, IPPT)			\
void									\
FCN(									\
  complex<T> const* A,							\
  T*                Z,							\
  length_type len)							\
{									\
  VSIP_IMPL_COVER_FCN("IPP_V_CR", IPPFCN)				\
  if (len > 0)								\
  {									\
    IppStatus status = IPPFCN(						\
      reinterpret_cast<IPPCT const*>(A),				\
      reinterpret_cast<IPPT*>(Z),					\
      static_cast<int>(len));						\
    assert(status == ippStsNoErr);					\
  }									\
}

#define VSIP_IMPL_IPP_VV(FCN, T, IPPFCN, IPPT)				\
void									\
FCN(									\
  T const* A,								\
  T const* B,								\
  T*       Z,								\
  length_type len)							\
{									\
  if (len > 0)								\
  {									\
    IppStatus status = IPPFCN(						\
      reinterpret_cast<IPPT const*>(A),					\
      reinterpret_cast<IPPT const*>(B),					\
      reinterpret_cast<IPPT*>(Z),					\
      static_cast<int>(len));						\
    assert(status == ippStsNoErr);					\
  }									\
}

#define VSIP_IMPL_IPP_VV_R(FCN, T, IPPFCN, IPPT)			\
void									\
FCN(									\
  T const* A,								\
  T const* B,								\
  T*       Z,								\
  length_type len)							\
{									\
  if (len > 0)								\
  {									\
    IppStatus status = IPPFCN(						\
      reinterpret_cast<IPPT const*>(B),					\
      reinterpret_cast<IPPT const*>(A),					\
      reinterpret_cast<IPPT*>(Z),					\
      static_cast<int>(len));						\
    assert(status == ippStsNoErr);					\
  }									\
}

// Copy
VSIP_IMPL_IPP_V(vcopy, float,           ippsCopy_32f,  Ipp32f)
VSIP_IMPL_IPP_V(vcopy, double,          ippsCopy_64f,  Ipp64f)
VSIP_IMPL_IPP_V(vcopy, complex<float>,  ippsCopy_32fc, Ipp32fc)
VSIP_IMPL_IPP_V(vcopy, complex<double>, ippsCopy_64fc, Ipp64fc)

// Zero
VSIP_IMPL_IPP_VGEN0(vzero, float,           ippsZero_32f,  Ipp32f)
VSIP_IMPL_IPP_VGEN0(vzero, double,          ippsZero_64f,  Ipp64f)
VSIP_IMPL_IPP_VGEN0(vzero, complex<float>,  ippsZero_32fc, Ipp32fc)
VSIP_IMPL_IPP_VGEN0(vzero, complex<double>, ippsZero_64fc, Ipp64fc)


// Convolution
void conv(float const *coeff, length_type coeff_size,
	  float const *in, length_type in_size,
	  float* out)
{
  ippsConv_32f(coeff, coeff_size, in, in_size, out);
}

void conv(double const *coeff, length_type coeff_size,
	  double const *in, length_type in_size,
	  double* out)
{
  ippsConv_64f(coeff, coeff_size, in, in_size, out);
}

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
	     length_type out_row_stride)
{
  IppiSize coeff_size = { coeff_cols, coeff_rows };
  IppiSize in_size    = { in_cols,    in_rows };
  
  ippiConvFull_16s_C1R(
    coeff, coeff_row_stride*sizeof(short), coeff_size,
    in,    in_row_stride   *sizeof(short),    in_size,
    out,   out_row_stride  *sizeof(short),
    1);
}

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
	     length_type out_row_stride)
{
  IppiSize coeff_size = { coeff_cols, coeff_rows };
  IppiSize in_size    = { in_cols,    in_rows };
  
  ippiConvFull_32f_C1R(
    coeff, coeff_row_stride*sizeof(float), coeff_size,
    in,    in_row_stride   *sizeof(float),    in_size,
    out,   out_row_stride  *sizeof(float));
}

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
	      length_type out_row_stride)
{
  IppiSize coeff_size = { coeff_cols, coeff_rows };
  IppiSize in_size    = { in_cols,    in_rows };
  
  ippiConvValid_32f_C1R(
    coeff, coeff_row_stride*sizeof(float), coeff_size,
    in,    in_row_stride   *sizeof(float),    in_size,
    out,   out_row_stride  *sizeof(float));
}

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
	      length_type out_row_stride)
{
  IppiSize coeff_size = { coeff_cols, coeff_rows };
  IppiSize in_size    = { in_cols,    in_rows };
  
  ippiConvValid_16s_C1R(
    coeff, coeff_row_stride*sizeof(short), coeff_size,
    in,    in_row_stride   *sizeof(short),    in_size,
    out,   out_row_stride  *sizeof(short),
    1);
}

} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip
