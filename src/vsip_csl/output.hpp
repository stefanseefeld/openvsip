/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/output.hpp
    @author  Jules Bergmann
    @date    2005-03-22
    @brief   VSIPL++ CodeSourcery Library: Output utilities.
*/

#ifndef VSIP_CSL_OUTPUT_HPP
#define VSIP_CSL_OUTPUT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>

#include <vsip_csl/output/domain.hpp>



namespace vsip_csl
{

/***********************************************************************
  Definitions
***********************************************************************/

/// Write a vector to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
operator<<(
  std::ostream&		       out,
  vsip::const_Vector<T, Block> vec)
  VSIP_NOTHROW
{
  for (vsip::index_type i=0; i<vec.size(); ++i)
    out << "  " << i << ": " << vec.get(i) << "\n";
  return out;
}



/// Write a matrix to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
operator<<(
  std::ostream&		       out,
  vsip::const_Matrix<T, Block> v)
  VSIP_NOTHROW
{
  for (vsip::index_type r=0; r<v.size(0); ++r)
  {
    out << "  " << r << ":";
    for (vsip::index_type c=0; c<v.size(1); ++c)
      out << "  " << v.get(r, c);
    out << std::endl;
  }
  return out;
}

/// Write a tensor to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
operator<<(
  std::ostream&		       out,
  vsip::const_Tensor<T, Block> v)
  VSIP_NOTHROW
{
  for (vsip::index_type z=0; z<v.size(0); ++z)
  {
    out << "plane " << z << ":\n";
    for (vsip::index_type r=0; r<v.size(1); ++r)
    {
      out << "  " << r << ":";
      for (vsip::index_type c=0; c<v.size(2); ++c)
        out << "  " << v.get(z, r, c);
      out << std::endl;
    }
    out << std::endl;
  }
  return out;
}

} // namespace vsip


// Declare this operator so that users do not need to include
// this directive, nor pull in the entire namespace.
using vsip_csl::operator<<;


#endif // VSIP_CSL_OUTPUT_HPP
