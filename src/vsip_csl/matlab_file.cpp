/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/matlab_file.cpp
    @author  Assem Salama
    @date    2006-06-21
    @brief   VSIPL++ CodeSourcery Library: Matlab_file class functions
*/

#include <vsip_csl/matlab_bin_formatter.hpp>
#include <vsip_csl/matlab_file.hpp>

namespace vsip_csl
{

Matlab_file::Matlab_file(std::string fname) :
  is_(fname.c_str()),
  begin_iterator_(false,this),
  end_iterator_(true,this)

{
  // check to make sure we successfully opened the file
  if (!is_)
    VSIP_IMPL_THROW(std::runtime_error(
      "Cannot open Matlab file '" + fname + "'"));

  // read header to make sure it is matlab file
  is_ >> matlab_header_;

  // get length of file
  {
    std::istream::off_type temp_offset = 0;
    std::istream::pos_type temp_pos = is_.tellg();
    is_.seekg(temp_offset,std::ios::end);
    length_ = static_cast<vsip::impl::uint32_type>(is_.tellg());
    is_.seekg(temp_pos);
  }
  view_header_.swap_bytes = matlab_header_.endian == ('I' << 8|'M');

  // read first header
  begin_iterator_.read_header();
  // set the end_of_file_ flag
  end_of_file_ = false;
  read_data_ = false;

}

}
