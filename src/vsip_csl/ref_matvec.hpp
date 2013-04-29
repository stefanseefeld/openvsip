/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ref_matvec.hpp
    @author  Jules Bergmann
    @date    2005-10-11
    @brief   VSIPL++ CodeSourcery Library: Reference implementations of 
             matvec routines.
*/

#ifndef VSIP_CSL_REF_MATVEC_HPP
#define VSIP_CSL_REF_MATVEC_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>

#include <vsip/support.hpp>
#include <vsip/vector.hpp>


namespace vsip_csl
{

/***********************************************************************
  Reference Definitions
***********************************************************************/

namespace ref
{

// Reference dot-product function.

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
typename vsip::Promotion<T0, T1>::type
dot(
  vsip::const_Vector<T0, Block0> u,
  vsip::const_Vector<T1, Block1> v)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;

  assert(u.size() == v.size());

  return_type sum = return_type();

  for (vsip::index_type i=0; i<u.size(); ++i)
    sum += u.get(i) * v.get(i);

  return sum;
}



// Reference outer-product functions.

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
vsip::Matrix<typename vsip::Promotion<T0, T1>::type>
outer(
  vsip::const_Vector<T0, Block0> u,
  vsip::const_Vector<T1, Block1> v)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;

  vsip::Matrix<return_type> r(u.size(), v.size());

  for (vsip::index_type i=0; i<u.size(); ++i)
    for (vsip::index_type j=0; j<v.size(); ++j)
      // r(i, j) = u(i) * v(j);
      r.put(i, j, u.get(i) * v.get(j));

  return r;
}

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
vsip::Matrix<typename vsip::Promotion<std::complex<T0>, std::complex<T1> >::type>
outer(
  vsip::const_Vector<std::complex<T0>, Block0> u,
  vsip::const_Vector<std::complex<T1>, Block1> v)
{
  typedef typename vsip::Promotion<std::complex<T0>, std::complex<T1> >::type return_type;

  vsip::Matrix<return_type> r(u.size(), v.size());

  for (vsip::index_type i=0; i<u.size(); ++i)
    for (vsip::index_type j=0; j<v.size(); ++j)
      // r(i, j) = u(i) * v(j);
      r.put(i, j, u.get(i) * conj(v.get(j)));

  return r;
}


// Reference vector-vector product

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
vsip::Matrix<typename vsip::Promotion<T0, T1>::type>
vv_prod(
  vsip::const_Vector<T0, Block0> u,
  vsip::const_Vector<T1, Block1> v)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;

  vsip::Matrix<return_type> r(u.size(), v.size());

  for (vsip::index_type i=0; i<u.size(); ++i)
    for (vsip::index_type j=0; j<v.size(); ++j)
      // r(i, j) = u(i) * v(j);
      r.put(i, j, u.get(i) * v.get(j));

  return r;
}




// Reference matrix-matrix product function (using vv-product).

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
vsip::Matrix<typename vsip::Promotion<T0, T1>::type>
prod(
  vsip::const_Matrix<T0, Block0> a,
  vsip::const_Matrix<T1, Block1> b)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;

  assert(a.size(1) == b.size(0));

  vsip::Matrix<return_type> r(a.size(0), b.size(1), return_type());

  for (vsip::index_type k=0; k<a.size(1); ++k)
    r += ref::vv_prod(a.col(k), b.row(k));

  return r;
}


// Reference matrix-vector product function (using dot-product).

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
vsip::Vector<typename vsip::Promotion<T0, T1>::type>
prod(
  vsip::const_Matrix<T0, Block0> a,
  vsip::const_Vector<T1, Block1> b)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;

  assert(a.size(1) == b.size(0));

  vsip::Vector<return_type> r(a.size(0), return_type());

  for (vsip::index_type k=0; k<a.size(0); ++k)
    r.put( k, ref::dot(a.row(k), b) );

  return r;
}


// Reference vector-matrix product function (using dot-product).

template <typename T0,
	  typename T1,
	  typename Block0,
	  typename Block1>
vsip::Vector<typename vsip::Promotion<T0, T1>::type>
prod(
  vsip::const_Vector<T1, Block1> a,
  vsip::const_Matrix<T0, Block0> b)
{
  typedef typename vsip::Promotion<T0, T1>::type return_type;

  assert(a.size(0) == b.size(0));

  vsip::Vector<return_type> r(b.size(1), return_type());

  for (vsip::index_type k=0; k<b.size(1); ++k)
    r.put( k, ref::dot(a, b.col(k)) );

  return r;
}



} // namespace ref
} // namespace vsip_csl

#endif // VSIP_CSL_REF_MATVEC_HPP
