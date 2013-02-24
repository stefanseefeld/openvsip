/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/matlab_text_formatter.hpp
    @author  Assem Salama
    @date    2006-05-22
    @brief   VSIPL++ CodeSourcery Library: Matlab text formatter
*/

#ifndef VSIP_CSL_MATLAB_TEXT_FORMATTER_HPP
#define VSIP_CSL_MATLAB_TEXT_FORMATTER_HPP

#include <string>
#include <vsip/support.hpp>

namespace vsip_csl
{

  /// This struct is just used as a wrapper so that we can overload the
  /// << operator
  template <typename ViewT>
  struct Matlab_text_formatter
  {
    Matlab_text_formatter(ViewT v) : v_(v), view_name_("a")  {}
    Matlab_text_formatter(ViewT v,std::string name) :
      v_(v), view_name_(name)  {}

    ViewT v_;
    std::string view_name_;
  };


} // namespace vsip_csl


/****************************************************************************
 * Definitions
 ***************************************************************************/

namespace vsip_csl
{

/// Write a matrix to a stream using a Matlab_text_formatter

template <typename T,
          typename Block0>
inline
std::ostream&
operator<<(
  std::ostream&		                                out,
  Matlab_text_formatter<vsip::Matrix<T,Block0> >        mf)
  VSIP_NOTHROW

{
  out << mf.view_name_ << " = " << std::endl;
  out << "[" << std::endl;
  for(vsip::index_type i=0;i<mf.v_.size(0);i++) {
    out << "  [ ";
    for(vsip::index_type j=0;j<mf.v_.size(1);j++)
      out << mf.v_.get(i,j) << " ";
    out << "]" << std::endl;
  }
  out << "];" << std::endl;

  return out;
}

/// Write a vector to a stream using a Matlab_text_formatter

template <typename T,
          typename Block0>
inline
std::ostream&
operator<<(
  std::ostream&		                          out,
  Matlab_text_formatter<vsip::Vector<T,Block0> >  mf)
  VSIP_NOTHROW

{
  out << mf.view_name_ << " = " << std::endl;
  out << "[ "; 
  for(vsip::index_type i=0;i<mf.v_.size(0);i++) {
    out << mf.v_.get(i) << " ";
  }
  out << "];" << std::endl;

  return out;
}

} // namespace vsip_csl

#endif // VSIP_CSL_MATLAB_TEXT_FORMATTER_HPP
