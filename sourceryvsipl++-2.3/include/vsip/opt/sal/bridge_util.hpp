/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/bridge_util.hpp
    @author  Jules Bergmann
    @date    2006-05-30
    @brief   VSIPL++ Library: Wrappers and traits to bridge with 
               Mercury SAL.
*/

#ifndef VSIP_OPT_SAL_BRIDGE_UTIL_HPP
#define VSIP_OPT_SAL_BRIDGE_UTIL_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <sal.h>

#include <vsip/support.hpp>
#include <vsip/core/coverage.hpp>
#include <vsip/core/storage.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace sal
{

template <typename T> struct Sal_split;
template <> struct Sal_split<float>  { typedef COMPLEX_SPLIT type; };
template <> struct Sal_split<double> { typedef DOUBLE_COMPLEX_SPLIT type; };

template <typename T> struct Sal_inter;
template <> struct Sal_inter<float>  { typedef COMPLEX type; };
template <> struct Sal_inter<double> { typedef DOUBLE_COMPLEX type; };


template <typename T,
	  typename ComplexT>
struct Sal_pointer
{ typedef T type; };

template <> struct Sal_pointer<complex<float>, Cmplx_split_fmt>
{ typedef COMPLEX_SPLIT type; };

template <> struct Sal_pointer<complex<double>, Cmplx_split_fmt>
{ typedef DOUBLE_COMPLEX_SPLIT type; };

template <> struct Sal_pointer<complex<float>, Cmplx_inter_fmt>
{ typedef COMPLEX type; };

template <> struct Sal_pointer<complex<double>, Cmplx_inter_fmt>
{ typedef DOUBLE_COMPLEX type; };



template <typename T,
	  typename ComplexT = Any_type>
struct Sal_vector
{
  // typedef typename Sal_pointer<T, ComplexT>::type type;
  typedef typename Storage<ComplexT, T>::type     type;

  type        ptr;
  stride_type stride;

  Sal_vector(type a_ptr, stride_type a_stride)
    : ptr(a_ptr), stride(a_stride)
  {}

  template <typename ExtT>
  explicit Sal_vector(ExtT /*const*/& ext)
    : ptr(ext.ext_.data()), stride(ext.ext_.stride(0))
  {}
};

template <typename T>
struct Sal_scalar
{
  T value;

  Sal_scalar(T a_value)
    : value(a_value) {}

  template <typename ExtWrapT>
  explicit Sal_scalar(ExtWrapT const& ext)
    : value(ext.value_) {}

};


} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SAL_BRIDGE_UTIL_HPP
