/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/output/domain.hpp
    @author  Jules Bergmann
    @date    2007-08-22
    @brief   VSIPL++ CodeSourcery Library: Output utilities for domains.
*/

#ifndef VSIP_CSL_OUTPUT_DOMAIN_HPP
#define VSIP_CSL_OUTPUT_DOMAIN_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/domain.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

/// Write a Domain<1> object to an output stream.

inline
std::ostream&
operator<<(
  std::ostream&		 out,
  vsip::Domain<1> const& dom)
  VSIP_NOTHROW
{
  out << "("
      << dom.first() << ","
      << dom.stride() << ","
      << dom.length() << ")";
  return out;
}



/// Write a Domain<2> object to an output stream.

inline
std::ostream&
operator<<(
  std::ostream&		 out,
  vsip::Domain<2> const& dom)
  VSIP_NOTHROW
{
  out << "(" << dom[0] << ", " << dom[1] << ")";
  return out;
}



/// Write a Domain<3> object to an output stream.

inline
std::ostream&
operator<<(
  std::ostream&		 out,
  vsip::Domain<3> const& dom)
  VSIP_NOTHROW
{
  out << "(" << dom[0] << ", " << dom[1] << ", " << dom[2] << ")";
  return out;
}



/// Write an Index to a stream.

template <vsip::dimension_type Dim>
inline
std::ostream&
operator<<(
  std::ostream&		        out,
  vsip::Index<Dim> const& idx)
  VSIP_NOTHROW
{
  out << "(";
  for (vsip::dimension_type d=0; d<Dim; ++d)
  {
    if (d > 0) out << ", ";
    out << idx[d];
  }
  out << ")";
  return out;
}



namespace impl
{

/// Write a Length to a stream.

template <vsip::dimension_type Dim>
inline
std::ostream&
operator<<(
  std::ostream&		         out,
  vsip::impl::Length<Dim> const& idx)
  VSIP_NOTHROW
{
  out << "(";
  for (vsip::dimension_type d=0; d<Dim; ++d)
  {
    if (d > 0) out << ", ";
    out << idx[d];
  }
  out << ")";
  return out;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CSL_OUTPUT_HPP
