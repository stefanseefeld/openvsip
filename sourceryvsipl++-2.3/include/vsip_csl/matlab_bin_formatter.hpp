/* Copyright (c) 2006, 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/matlab_bin_formatter.hpp
    @author  Assem Salama
    @date    2006-05-22
    @brief   VSIPL++ CodeSourcery Library: Matlab binary formatter
*/

/*  This file is written to conform to the Matlab .mat file specification,
    downloaded from this url on 2009-04-23:
    http://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf
*/

#ifndef VSIP_CSL_MATLAB_BIN_FORMATTER_HPP
#define VSIP_CSL_MATLAB_BIN_FORMATTER_HPP

#include <sstream>
#include <string>
#include <limits>
#include <cstring>

#include <vsip_csl/matlab.hpp>
#include <vsip/core/fns_scalar.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip/core/extdata.hpp>

namespace vsip_csl
{

  struct Matlab_bin_hdr
  {
    Matlab_bin_hdr(std::string const& user_descr)
      : description     ("MATLAB 5.0 : "),
	version         (0x100)
    {
      description.append(user_descr);
    }

    Matlab_bin_hdr()
      : description     ("MATLAB 5.0 : "),
	version         (0x100)
    {}

    // description
    std::string             description;
    vsip::impl::uint16_type version;
    vsip::impl::uint16_type endian;

  };

  template <typename ViewT>
  struct Matlab_bin_formatter
  {
    Matlab_bin_formatter(ViewT v,std::string const& name) :
      view(v), name(name), header()  {}
    Matlab_bin_formatter(ViewT v,std::string const& name,
      Matlab_bin_hdr &h) :
        view(v), name(name), header(h)  {}

    ViewT view;
    std::string name;
    Matlab_bin_hdr header;

  };

  struct Matlab_view_header
  {
    bool                    swap_bytes;
    char                    array_name[128];
    bool                    is_complex;
    vsip::impl::uint8_type  class_type;
    vsip::impl::uint32_type num_dims;
    vsip::impl::uint32_type dims[3]; // max dimensions
    std::istream::pos_type  next_header;
  };


} // namespace vsip_csl

/****************************************************************************
 * Definitions
 ***************************************************************************/

namespace vsip_csl
{

// Write matlab file header.
inline
std::ostream&
operator<<(
  std::ostream&           o,
  Matlab_bin_hdr const&   h)
{
  matlab::File_header m_hdr;

  // set hdr to spaces
  memset(&(m_hdr),' ',sizeof(m_hdr));
  strncpy(m_hdr.description, h.description.data(), h.description.length());
  m_hdr.version = 0x0100;
  m_hdr.endian = 'M' << 8 | 'I';

  // write header
  o.write(reinterpret_cast<char*>(&m_hdr),sizeof(m_hdr));

  return o;
}
// operator to write a view to a matlab file
template <typename T,
          typename Block0,
	  template <typename,typename> class const_View>
inline
std::ostream&
operator<<(
  std::ostream&                                       o,
  Matlab_bin_formatter<const_View<T,Block0> > const&  mbf)
{
  typedef typename vsip::impl::Scalar_of<T>::type scalar_type;
  matlab::data_element temp_data_element;
  size_t    sz;
  matlab::view_header<vsip::impl::Dim_of_view<const_View>::dim > m_view;
  vsip::length_type num_points = mbf.view.size();
  vsip::dimension_type v_dims = vsip::impl::Dim_of_view<const_View>::dim;

  memset(&m_view,0,sizeof(m_view));

  // matrix data type
  m_view.header.type = matlab::miMATRIX;
  m_view.header.size = 1; // TEMP

  // array flags
  m_view.array_flags_header.type = matlab::miUINT32;
  m_view.array_flags_header.size = 8;
  if(vsip::impl::Is_complex<T>::value) 
    m_view.array_flags[0] |= (1<<11); // Complex

  // fill in class
  m_view.array_flags[0] |= 
    (matlab::Matlab_header_traits<sizeof(scalar_type),
                  std::numeric_limits<scalar_type>::is_signed,
                  std::numeric_limits<scalar_type>::is_integer>::class_type);

  // dimension sizes
  m_view.dim_header.type = matlab::miINT32;

  // fill in dimensions
  if(v_dims == 1)
  {
    // Matlab stores vectors as 2D arrays where one of the dimensions
    // is 1.  We make the first dimension a 1, thereby saving Vector
    // views as row vectors rather than column vectors.  (Most Matlab
    // code creates row vectors unless it is explicitly creating a
    // column vector.)
    m_view.dim_header.size = 2*4; // 4 bytes per dimension
    m_view.dim[0] = 1;
    m_view.dim[1] = mbf.view.size(0);
  }
  else
  {
    m_view.dim_header.size = v_dims*4; // 4 bytes per dimension
    for(vsip::dimension_type i =0;i<v_dims;i++)
    {
      m_view.dim[i] = mbf.view.size(i);
    }
  }
  // array name
  m_view.array_name_header.type = matlab::miINT8;
  m_view.array_name_header.size = mbf.name.length();


  // calculate size
  sz = sizeof(m_view)-8;
  sz += mbf.name.length();
  sz += (8-mbf.name.length())&0x7;
  sz += 8; // 8 bytes of header for real data
  if(vsip::impl::Is_complex<T>::value) sz += 8; // 8 more for complex data
  sz += num_points*sizeof(T);
  m_view.header.size = sz;

  o.write(reinterpret_cast<char*>(&m_view),sizeof(m_view));

  // write array name
  o.write(mbf.name.c_str(),mbf.name.length());
  // pad
  { 
    char c=0;
    for(vsip::length_type i=0;i<((8-mbf.name.length())&0x7);i++) o.write(&c,1);
  }

  // write data
  {
  
    // make sure we don't need a copy if we use Ext data
    // This code block is disabled with '&& false' until we figure out how to
    // make it reliably produce column-major data and fix the padding.
    if(vsip::impl::Ext_data_cost<Block0,
      typename matlab::Matlab_desired_LP<const_View>::type >::value==0 && false)
    {
      vsip::impl::Ext_data<Block0,
	                 typename matlab::Matlab_desired_LP<const_View>::type >
	     
	       m_ext(mbf.view.block());

      typedef typename vsip::impl::Ext_data<Block0,
	typename matlab::Matlab_desired_LP<const_View>::type >::storage_type
		storage_type;

      temp_data_element.type = matlab::Matlab_header_traits<sizeof(scalar_type),
                  std::numeric_limits<scalar_type>::is_signed,
                  std::numeric_limits<scalar_type>::is_integer>::value_type;

      temp_data_element.size = num_points*sizeof(scalar_type);
      for(int i=0;i<=vsip::impl::Is_complex<T>::value;i++)
      {
        o.write(reinterpret_cast<char*>(&temp_data_element),
                  sizeof(temp_data_element));
        if(i==0) o.write(reinterpret_cast<char*>
             (storage_type::get_real_ptr(m_ext.data())),
                  num_points*sizeof(scalar_type));
        else o.write(reinterpret_cast<char*>
             (storage_type::get_imag_ptr(m_ext.data())),
                  num_points*sizeof(scalar_type));
      }
    }
    else
    {
      typedef matlab::Subview_helper<const_View<T,Block0> > subview;
      typedef typename subview::realview_type r_v;
      typedef typename subview::imagview_type i_v;

      // conventional way
      temp_data_element.type = matlab::Matlab_header_traits<sizeof(scalar_type),
                  std::numeric_limits<scalar_type>::is_signed,
                  std::numeric_limits<scalar_type>::is_integer>::value_type;

      temp_data_element.size = num_points*sizeof(scalar_type);
      for(int i=0;i<=vsip::impl::Is_complex<T>::value;i++)
      {
        o.write(reinterpret_cast<char*>(&temp_data_element),
                  sizeof(temp_data_element));
        if(i==0) matlab::write<r_v>(o,subview::real(mbf.view));
        else     matlab::write<i_v>(o,subview::imag(mbf.view));
      }
    }
  }

  return o;
}



// Read matlab file header.
inline
std::istream&
operator>>(
  std::istream&           is,
  Matlab_bin_hdr&         h)
{
  matlab::File_header m_hdr;

  // read header
  is.read(reinterpret_cast<char*>(&m_hdr),sizeof(m_hdr));
  if(is.gcount() < static_cast<std::streamsize>(sizeof(m_hdr)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_bin_hdr: Unexpected end of file"));

  m_hdr.description[matlab::File_header::description_size-1] = '\0';
  h.description = std::string(m_hdr.description);

  h.version = m_hdr.version;
  h.endian  = m_hdr.endian;

  return is;
}



inline void
skip_padding(std::istream& is, vsip::length_type length)
{
  is.ignore((8-length)&0x7);
}



// Read a matlab view header into a Matlab_view_header class
//
// On disk:
//   |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |
//   |-----+-----+-----+-----+-----+-----+-----+-----|
//   |  miMATRIX             |  <size>               |  < data element header
//
//   |  miUINT32             |      8                |  < ARRAY FLAGS
//   |  undef    | <F> | <C> |  undef                | 
//
//   |  miUINT32             |      12               | < DIMENSIONS ARRAY
//   |  dim0                 |      dim1             |   (3-dim shown)
//   |  dim2                 |      padding          | 

//   |        3  | miINT8    | 'a' | 'r' | 'r' | pad | < ARRAY NAME
//                                                       
//
//   Array Data.
//
// Notes:
//   <C> is the class_type, i.e. mxDOUBLE_CLASS, etc.
//
inline
std::istream&
operator>>(
  std::istream&           is,
  Matlab_view_header&     h)
{
  vsip::impl::uint32_type array_flags[2];
  vsip::impl::uint32_type dims[3];
  matlab::data_element temp_element;
  bool swap_bytes;
  typedef vsip::index_type index_type;
  typedef vsip::length_type length_type;

  swap_bytes = h.swap_bytes;

  // 1. Read overall data element header for the object.  Identifies:
  //  - type of data element (we can only handle miMATRIX),
  //  - overall size of data element.
  is.read(reinterpret_cast<char*>(&temp_element),sizeof(temp_element));
  if(is.gcount() < static_cast<std::streamsize>(sizeof(temp_element)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_view_header(read): Unexpected end of file"));
  matlab::swap<vsip::impl::int32_type> (&(temp_element.type),swap_bytes);
  matlab::swap<vsip::impl::uint32_type>(&(temp_element.size),swap_bytes);

  if (temp_element.type != matlab::miMATRIX)
  {
    std::ostringstream msg;
    msg << "Matlab_view_header(read): Unsupported data_element type: ";
    msg << "got " << matlab::data_type(temp_element.type)
	<< " (" << temp_element.type << ")"
	<< ", can only handl miMATRIX (" << matlab::miMATRIX << ")";
    VSIP_IMPL_THROW(std::runtime_error(msg.str()));
  }

  // store the file position of next header
  {
    std::istream::pos_type curr_pos = is.tellg();
    curr_pos               += temp_element.size;
    h.next_header          = curr_pos;
  }


  // 2. Read the array_flags.
  is.read(reinterpret_cast<char*>(&temp_element),sizeof(temp_element));
  if(is.gcount() < static_cast<std::streamsize>(sizeof(temp_element)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_view_header(read): Unexpected end of file"));
  matlab::swap<vsip::impl::int32_type> (&(temp_element.type),swap_bytes);
  matlab::swap<vsip::impl::uint32_type>(&(temp_element.size),swap_bytes);

  if (temp_element.type != matlab::miUINT32)
  {
    std::ostringstream msg;
    msg << "Matlab_view_header(read): Unexpected type for array flags: ";
    msg << "got " << temp_element.type << " bytes, expected miUINT32 (6)";
    VSIP_IMPL_THROW(std::runtime_error(msg.str()));
  }
  if (temp_element.size > 8)
  {
    std::ostringstream msg;
    msg << "Matlab_view_header(read): Length of array flags is too large: ";
    msg << "got " << temp_element.size << " bytes, expected " << 8 << " bytes";
    VSIP_IMPL_THROW(std::runtime_error(msg.str()));
  }
  if(temp_element.size > 8)
    VSIP_IMPL_THROW(std::runtime_error(
      "Length of array flags is too large"));
  is.read(reinterpret_cast<char*>(&array_flags),temp_element.size);
  if(is.gcount() < static_cast<std::streamsize>(temp_element.size))
    VSIP_IMPL_THROW(std::runtime_error(
     "Matlab_view_header(read): Unexpected end of file reading array flags"));
  for(index_type i=0;i<temp_element.size/4;i++)
    matlab::swap<vsip::impl::uint32_type>(&(array_flags[i]),swap_bytes);

  // is this complex?
  h.is_complex  = ((array_flags[0]&(1<<11)) == 1);
  h.class_type  = (array_flags[0]&0xff);


  // 3. Read dimensions.
  is.read(reinterpret_cast<char*>(&temp_element),sizeof(temp_element));
  if(is.gcount() < static_cast<std::streamsize>(sizeof(temp_element)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_view_header(read): Unexpected end of file reading dimensions (1)"));
  matlab::swap<vsip::impl::int32_type> (&(temp_element.type),swap_bytes);
  matlab::swap<vsip::impl::uint32_type>(&(temp_element.size),swap_bytes);

  if ((temp_element.type & 0x0000ffff) != matlab::miINT32)
  {
    std::ostringstream msg;
    msg << "Matlab_view_header(read): Unexpected type for dimensions array: ";
    msg << "got " << temp_element.type << ", expected miINT32 (5)";
    VSIP_IMPL_THROW(std::runtime_error(msg.str()));
  }
  if (temp_element.size > 12)
  {
    std::ostringstream msg;
    msg << "Matlab_view_header(read): Number of dimensions is too large: ";
    msg << "got " << temp_element.size/4 << ", can only handle 3";
    VSIP_IMPL_THROW(std::runtime_error(msg.str()));
  }

  is.read(reinterpret_cast<char*>(&dims),temp_element.size);
  if(is.gcount() < static_cast<std::streamsize>(temp_element.size))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_view_header(read): Unexpected end of file reading dimensions (2)"));
  skip_padding(is, temp_element.size);

  h.num_dims = temp_element.size/4;
  for(index_type i=0;i<temp_element.size/4;i++)
  {
    matlab::swap<vsip::impl::uint32_type>(&(dims[i]),swap_bytes);
    h.dims[i] = dims[i];
  }


  // 4. Read array name.
  is.read(reinterpret_cast<char*>(&temp_element),sizeof(temp_element));
  if(is.gcount() < static_cast<std::streamsize>(sizeof(temp_element)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_view_header(read): Unexpected end of file reading array name (1)"));
  matlab::swap<vsip::impl::int32_type>(&(temp_element.type),swap_bytes);
  // Don't swab the length yet, it may be a string.

  if ((temp_element.type & 0x0000ffff) != matlab::miINT8)
  {
    std::ostringstream msg;
    msg << "Matlab_view_header(read): Unexpected type for array name: ";
    msg << "got " << matlab::data_type(temp_element.type)
	<< " (" << temp_element.type << "),"
	<< " expected miINT8 (" << matlab::miINT8 << ")";
    VSIP_IMPL_THROW(std::runtime_error(msg.str()));
  }

  if(temp_element.type & 0xffff0000)
  {
    int length = (temp_element.type & 0xffff0000) >> 16;
    // array name is short
    strncpy(h.array_name, reinterpret_cast<char*>(&temp_element.size), length);
    h.array_name[length] = 0;
  }
  else
  {
    matlab::swap<vsip::impl::uint32_type>(&(temp_element.size),swap_bytes);
    int length = temp_element.size;
    // the name is longer than 4 bytes
    if(length > 128)
      VSIP_IMPL_THROW(std::runtime_error("Name of view is too large"));

    is.read(h.array_name,length);
    if(is.gcount() < length)
      VSIP_IMPL_THROW(std::runtime_error(
	"Matlab_view_header(read): Unexpected end of file reading array name (2)"));
    h.array_name[length] = 0;
    skip_padding(is, length);
  }

  return is;
}



// operator to read view from matlab file
template <typename T,
          typename Block0,
	  template <typename,typename> class View>
inline
std::istream&
operator>>(
  std::istream&                                       is,
  Matlab_bin_formatter<View<T,Block0> >               mbf)
{
  matlab::data_element temp_data_element;
  matlab::view_header<vsip::impl::Dim_of_view<View>::dim> m_view;
  typedef typename vsip::impl::Scalar_of<T>::type scalar_type;
  typedef matlab::Subview_helper<View<T,Block0> > subview;
  typedef typename subview::realview_type r_v;
  typedef typename subview::imagview_type i_v;
  vsip::dimension_type v_dim = vsip::impl::Dim_of_view<View>::dim;
  vsip::impl::uint16_type endian = mbf.header.endian;
  bool swap_value;

  if(endian == ('I'<<8 | 'M')) swap_value = true;
  else if(endian == ('M'<<8 | 'I')) swap_value = false;
  else 
    VSIP_IMPL_THROW(std::runtime_error(
      "Bad endian field in matlab file"));


  // read header
  is.read(reinterpret_cast<char*>(&m_view),sizeof(m_view));
  if(is.gcount() < static_cast<std::streamsize>(sizeof(m_view)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Matlab_bin_formatter(read): Unexpected end of file (1)"));

  // do we need to swap fields?
  matlab::swap_header(m_view,swap_value);

  // are the data types compatible?
  if(!vsip::impl::Is_complex<T>::value && (m_view.array_flags[0]&(1<<11)))
    VSIP_IMPL_THROW(std::runtime_error(
      "Trying to read complex data into a real view"));

  if(!matlab::class_type_is_int(m_view.array_flags[0] & 0xff)
      && std::numeric_limits<scalar_type>::is_integer)
    VSIP_IMPL_THROW(std::runtime_error(
      "Trying to read float data into an integer view"));

  if(matlab::class_type_is_signed(m_view.array_flags[0] & 0xff)
      && !std::numeric_limits<scalar_type>::is_signed)
    VSIP_IMPL_THROW(std::runtime_error(
      "Trying to read signed data into an unsigned view"));

  // do dimensions agree?
  if(v_dim == 1)
  {
    // Matlab stores vectors as either Nx1 or 1xN matrices, depending
    // on whether they are column vectors or row vectors.  We do not
    // distinguish the two types, and thus must handle both of them.
    if((m_view.dim_header.size/4) != 2)
      VSIP_IMPL_THROW(std::runtime_error(
        "Trying to read a view of different dimensions"));
    
    if(m_view.dim[0] == 1) // Row vector
    {
      if(mbf.view.size(0) != m_view.dim[1])
        VSIP_IMPL_THROW(std::runtime_error(
          "View dimensions don't agree"));
    }
    else if(m_view.dim[1] == 1) // Column vector
    {
      if(mbf.view.size(0) != m_view.dim[0])
        VSIP_IMPL_THROW(std::runtime_error(
          "View dimensions don't agree"));
    }
    else
    {
      VSIP_IMPL_THROW(std::runtime_error(
        "Trying to read a view of different dimensions"));
    }
  }
  else
  {
    if(v_dim != (m_view.dim_header.size/4))
      VSIP_IMPL_THROW(std::runtime_error(
        "Trying to read a view of different dimensions"));

    for(vsip::dimension_type i=0;i<v_dim;i++)
      if(mbf.view.size(i) != m_view.dim[i])
        VSIP_IMPL_THROW(std::runtime_error(
          "View dimensions don't agree"));
  }

  // read array name
  if(m_view.array_name_header.type & 0xffff0000)
  {
    // array name is short

  }
  else
  {
    int length = m_view.array_name_header.size;
    char c;
    char c_array[128];
    // the name is longer than 4 bytes
    if(length > 128)
      VSIP_IMPL_THROW(std::runtime_error(
        "Name of view is too large"));

    is.read(c_array,length);
    c_array[length] = 0;
    // read padding
    for(int i=0;i<((8-length)&0x7);i++) is.read(&c,1);
  }

  // read data, we will go in this loop twice if we have complex data
  
  int complex_data;
  if (m_view.array_flags[0]&(1<<11))
    complex_data=1;
  else
    complex_data=0;
  
  for (int i=0;i <= complex_data;i++)
  {

    // read data header
    is.read(reinterpret_cast<char*>(&temp_data_element),
            sizeof(temp_data_element));
    if(is.gcount() < static_cast<std::streamsize>(sizeof(temp_data_element)))
      VSIP_IMPL_THROW(std::runtime_error(
        "Matlab_bin_formatter(read): Unexpected end of file (2)"));

    // should we swap this field?
    matlab::swap<vsip::impl::int32_type>(&(temp_data_element.type),swap_value);
    matlab::swap<vsip::impl::uint32_type>(&(temp_data_element.size),swap_value);


    // Because we don't know how the data was stored, we need to instantiate
    // generic_reader which can read a type and cast into a different one
    if(temp_data_element.type == matlab::miINT8) 
    {
      if(i==0)
        matlab::read<vsip::impl::int8_type, r_v>(is,
                                                 subview::real(mbf.view),
                                                 swap_value);
      else
        matlab::read<vsip::impl::int8_type, i_v>(is,
                                                 subview::imag(mbf.view),
                                                 swap_value);
    }
    else if(temp_data_element.type == matlab::miUINT8) 
    {
      if(i==0)
        matlab::read<vsip::impl::uint8_type, r_v>(is,
                                                  subview::real(mbf.view),
                                                  swap_value);
      else  
        matlab::read<vsip::impl::uint8_type, i_v>(is,
                                                  subview::imag(mbf.view),
                                                  swap_value);
    }
    else if(temp_data_element.type == matlab::miINT16) 
    {
      if(i==0)
        matlab::read<vsip::impl::int16_type, r_v>(is,
                                                  subview::real(mbf.view),
                                                  swap_value);
      else
        matlab::read<vsip::impl::int16_type, i_v>(is,
                                                  subview::imag(mbf.view),
                                                  swap_value);
    }
    else if(temp_data_element.type == matlab::miUINT16) 
    {
      if(i==0)
        matlab::read<vsip::impl::uint16_type, r_v>(is,
                                                   subview::real(mbf.view),
                                                   swap_value);
      else
        matlab::read<vsip::impl::uint16_type, i_v>(is,
                                                   subview::imag(mbf.view),
                                                   swap_value);
    }
    else if(temp_data_element.type == matlab::miINT32) 
    {
      if(i==0)
        matlab::read<vsip::impl::int32_type, r_v>(is,
                                                  subview::real(mbf.view),
                                                  swap_value);
      else
        matlab::read<vsip::impl::int32_type, i_v>(is,
                                                  subview::imag(mbf.view),
                                                  swap_value);
    }
    else if(temp_data_element.type == matlab::miUINT32) 
    {
      if(i==0)
        matlab::read<vsip::impl::uint32_type, r_v>(is,
                                                   subview::real(mbf.view),
                                                   swap_value);
      else
        matlab::read<vsip::impl::uint32_type, i_v>(is,
                                                   subview::imag(mbf.view),
                                                   swap_value);
    }
    else if(temp_data_element.type == matlab::miSINGLE) 
    {
      if(i==0)
        matlab::read<float, r_v>(is, subview::real(mbf.view), swap_value);
      else
        matlab::read<float, i_v>(is, subview::imag(mbf.view), swap_value);
    }
    else
    {
      if(i==0)
        matlab::read<double, r_v>(is, subview::real(mbf.view), swap_value);
      else
        matlab::read<double, i_v>(is, subview::imag(mbf.view), swap_value);
    }

  }

  // If loading real data into complex view, set imaginary parts to zero.
  if(vsip::impl::Is_complex<T>::value && !(m_view.array_flags[0]&(1<<11)))
    subview::imag(mbf.view) = 0;

  return is;
}



} // namespace vsip_csl

#endif // VSIP_CSL_MATLAB_BIN_FORMATTER_HPP
