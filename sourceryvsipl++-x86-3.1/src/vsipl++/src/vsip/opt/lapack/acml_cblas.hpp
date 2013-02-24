/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/lapack/acml_cblas.hpp
    @author  Jules Bergmann
    @date    2005-08-19
    @brief   VSIPL++ Library: ACML CBLAS wrappers.

    ACML doesn't provide CBLAS bindings.  Also, its headers define a
    complex type that is compatible with std::complex, but has the
    same name.  This file provides CBLAS bindings to ACML.
*/

#ifndef VSIP_OPT_LAPACK_ACML_CBLAS_HPP
#define VSIP_OPT_LAPACK_ACML_CBLAS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

extern "C"
{

extern float sdot(int const n, float const *x, int const incx, float const *y, int const incy);
extern double ddot(int const n, double const *x, int const incx, double const *y, int const incy);

extern std::complex<float> cdotc(int const n, void const *x, int const incx, void const *y, int const incy);
extern std::complex<float> cdotu(int const n, void const *x, int const incx, void const *y, int const incy);
extern std::complex<double> zdotc(int const n, void const *x, int const incx, void const *y, int const incy);
extern std::complex<double> zdotu(int const n, void const *x, int const incx, void const *y, int const incy);

} // extern "C"

float inline
cblas_sdot(
  const int    n,
  const float* x,
  const int    incx,
  const float* y,
  const int    incy)
{
  return sdot(n, x, incx, y, incy);
}

double inline
cblas_ddot(
  const int     n,
  const double* x,
  const int     incx,
  const double* y,
  const int     incy)
{
  return ddot(n, x, incx, y, incy);
}

void inline
cblas_cdotu_sub(
  int                 const  n,
  void const* x,
  int                 const  incx,
  void const* y,
  int                 const  incy,
  void*       dotu)
{
  *reinterpret_cast<std::complex<float>*>(dotu) = cdotu(n, x, incx, y, incy);
}

void inline
cblas_cdotc_sub(
  int                 const  n,
  void const* x,
  int                 const  incx,
  void const* y,
  int                 const  incy,
  void*       dotc)
{
  *reinterpret_cast<std::complex<float>*>(dotc) = cdotc(n, x, incx, y, incy);
}

void inline
cblas_zdotu_sub(
  int                  const  n,
  void const* x,
  int                  const  incx,
  void const* y,
  int                  const  incy,
  void*       dotu)
{
  *reinterpret_cast<std::complex<double>*>(dotu) = zdotu(n, x, incx, y, incy);
}

void inline
cblas_zdotc_sub(
  int                  const  n,
  void const* x,
  int                  const  incx,
  void const* y,
  int                  const  incy,
  void*       dotc)
{
  *reinterpret_cast<std::complex<double>*>(dotc) = zdotc(n, x, incx, y, incy);
}

#endif // VSIP_OPT_LAPACK_ACML_CBLAS_HPP
