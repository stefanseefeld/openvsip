/* Copyright (c) 2006, 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_CSL_MATLAB_HPP
#define VSIP_CSL_MATLAB_HPP

#include <vsip/support.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/inttypes.hpp>
#include <iostream>

namespace vsip_csl
{

namespace matlab
{
  struct data_element
  {
    vsip::impl::int32_type type;
    vsip::impl::uint32_type size;
  };

  template <vsip::dimension_type Dim>
  struct view_header
  {
    data_element header;
    data_element array_flags_header;
    vsip::impl::uint32_type array_flags[2];
    data_element dim_header;
    // The dimension has to be aligned to an 8 byte boundary.  Also,
    // Matlab stores vectors as 2D arrays, so Dim of 1 needs dim[2].
    vsip::impl::uint32_type dim[Dim + Dim%2];
    data_element array_name_header;
  };

  // Helper struct to get the imaginary part of a view.
  template <typename ViewT,
            bool IsComplex =
	      vsip::impl::Is_complex<typename ViewT::value_type>::value>
  struct Subview_helper;

  template <typename ViewT>
  struct Subview_helper<ViewT,true>
  {
    typedef typename ViewT::realview_type realview_type;
    typedef typename ViewT::imagview_type imagview_type;

    static realview_type real(ViewT v) { return v.real(); }
    static imagview_type imag(ViewT v) { return v.imag(); }
  };

  template <typename ViewT>
  struct Subview_helper<ViewT,false>
  {
    typedef ViewT realview_type;
    typedef ViewT imagview_type;

    static realview_type real(ViewT v) { return v; }
    static imagview_type imag(ViewT v) { return v; }
  };

  template <typename T = char,
	    bool to_swap_or_not_to_swap = false,
            size_t type_size = sizeof(T),
            bool IsComplex = vsip::impl::Is_complex<T>::value>
  struct Swap_value 
  { 
    static void swap(T *d) {d=d;} 
  };

  template <typename T>
  struct Swap_value<T,true,2,false>
  {
    static void swap(T* d)
    {
      char *p = reinterpret_cast<char*>(d);
      std::swap(p[0],p[1]);
    }
  };

  template <typename T>
  struct Swap_value<T,true,4,false>
  {
    static void swap(T* d)
    {
      char *p = reinterpret_cast<char*>(d);
      std::swap(p[0],p[3]);
      std::swap(p[1],p[2]);
    }
  };

  template <typename T>
  struct Swap_value<T,true,8,false>
  {
    static void swap(T* d)
    {
      char *p = reinterpret_cast<char*>(d);
      std::swap(p[0],p[7]);
      std::swap(p[1],p[6]);
      std::swap(p[2],p[5]);
      std::swap(p[3],p[4]);
    }
  };

  template <typename T>
  struct Swap_value<T,true,8,true>   // complex
  {
    static void swap(T* d)
    {
      char *p = reinterpret_cast<char*>(d);
      std::swap(p[0],p[3]);
      std::swap(p[1],p[2]);
      std::swap(p[4],p[7]);
      std::swap(p[5],p[6]);
    }
  };

  template <typename T>
  struct Swap_value<T,true,16,true>  // complex
  {
    static void swap(T* d)
    {
      char *p = reinterpret_cast<char*>(d);
      std::swap(p[0],p[7]);
      std::swap(p[1],p[6]);
      std::swap(p[2],p[5]);
      std::swap(p[3],p[4]);
      std::swap(p[8],p[15]);
      std::swap(p[9],p[14]);
      std::swap(p[10],p[13]);
      std::swap(p[11],p[12]);
    }
  };


  // a swap wrapper function
  template <typename T>
  void swap(T *data, bool swap_bytes)
  {
    if(swap_bytes)
      Swap_value<T,true>::swap(data);
  }

  // swaps the header of a view
  template <vsip::dimension_type dim>
  void swap_header(view_header<dim> &header, bool swap_bytes)
  {
    if(swap_bytes)
    {
      typedef vsip::impl::int32_type int32_type;
      typedef vsip::impl::uint32_type uint32_type;
      // swap all fields
      Swap_value<int32_type,true>::swap(&(header.header.type));
      Swap_value<uint32_type,true>::swap(&(header.header.size));
      Swap_value<int32_type,true>::swap(&(header.array_flags_header.type));
      Swap_value<uint32_type,true>::swap(&(header.array_flags_header.size));
      Swap_value<int32_type,true>::swap(&(header.dim_header.type));
      Swap_value<uint32_type,true>::swap(&(header.dim_header.size));
      Swap_value<int32_type,true>::swap(&(header.array_name_header.type));
      Swap_value<uint32_type,true>::swap(&(header.array_name_header.size));
      for(vsip::index_type i=0;i<dim;i++)
        Swap_value<uint32_type,true>::swap(&(header.dim[i]));
      if (dim==1)
        Swap_value<uint32_type,true>::swap(&(header.dim[1]));
      for(vsip::index_type i=0;i<2;i++)
        Swap_value<uint32_type,true>::swap(&(header.array_flags[i]));
    }
  }

  // generic reader that allows us to read a generic type and cast to another
  
  // the read function for real or complex depending of the view that was
  // passed in
  template <typename T1,
	    typename ViewT>
  void read(std::istream& is,ViewT v,bool swap_bytes)
  {
    vsip::dimension_type const View_dim = ViewT::dim;
    vsip::Index<View_dim> my_index;
    vsip::impl::Length<View_dim> v_extent = extent(v);
    T1 data;
    typedef typename ViewT::value_type scalar_type;

    // get num_points
    vsip::length_type num_points = v.size();

    // read all the points
    for(vsip::index_type i=0;i<num_points;i++) {
      is.read(reinterpret_cast<char*>(&data),sizeof(data));
      swap(&data,swap_bytes);
      put(v,my_index,scalar_type(data));

      // increment index
      my_index = vsip::impl::next<typename vsip::impl::Col_major<View_dim>::type>
        (v_extent,my_index);
    }
    
    // Matlab data blocks must be padded to a multiple of 8 bytes.
    if (sizeof(data) < 8)
    {
      for(vsip::index_type i = 0; i < (num_points % 8/sizeof(data)); i++) {
        is.read(reinterpret_cast<char*>(&data),sizeof(data));
      }
    }
  }

  // a write function to output a view to a matlab file.
  template <typename ViewT>
  void write(std::ostream& os,ViewT v)
  {
    vsip::dimension_type const View_dim = ViewT::dim;
    vsip::Index<View_dim> my_index;
    vsip::impl::Length<View_dim> v_extent = extent(v);
    typename ViewT::value_type data;

    // get num_points
    vsip::length_type num_points = v.size();

    // write all the points
    for(vsip::index_type i=0;i<num_points;i++) {
      data = get(v,my_index);
      os.write(reinterpret_cast<char*>(&data),sizeof(data));

      // increment index
      my_index = vsip::impl::next<typename vsip::impl::Col_major<View_dim>::type>
        (v_extent,my_index);
    }

    // Matlab data blocks must be padded to a multiple of 8 bytes.
    if (sizeof(data) < 8)
    {
      for(vsip::index_type i = 0; i < (num_points % 8/sizeof(data)); i++) {
        os.write(reinterpret_cast<char*>(&data),sizeof(data));
      }
    }
  }

struct File_header
{
  static vsip::length_type const description_size = 116;
  static vsip::length_type const subsys_data_size = 8;

  char description[description_size];
  char subsys_data[subsys_data_size];
  vsip::impl::uint16_type version;
  vsip::impl::uint16_type endian;
};

  // constants for matlab binary format

  // data types
  static int const miINT8           = 1;
  static int const miUINT8          = 2;
  static int const miINT16          = 3;
  static int const miUINT16         = 4;
  static int const miINT32          = 5;
  static int const miUINT32         = 6;
  static int const miSINGLE         = 7;
  static int const miDOUBLE         = 9;
  static int const miINT64          = 12;
  static int const miUINT64         = 13;
  static int const miMATRIX         = 14;
  static int const miCOMPRESSED     = 15;
  static int const miUTF8           = 16;
  static int const miUTF16          = 17;
  static int const miUTF32          = 18;
  
  // class types
  static int const mxCELL_CLASS     = 1;
  static int const mxSTRUCT_CLASS   = 2;
  static int const mxOBJECT_CLASS   = 3;
  static int const mxCHAR_CLASS     = 4;
  static int const mxSPARSE_CLASS   = 5;
  static int const mxDOUBLE_CLASS   = 6;
  static int const mxSINGLE_CLASS   = 7;
  static int const mxINT8_CLASS     = 8;
  static int const mxUINT8_CLASS    = 9;
  static int const mxINT16_CLASS    = 10;
  static int const mxUINT16_CLASS   = 11;
  static int const mxINT32_CLASS    = 12;
  static int const mxUINT32_CLASS   = 13;

  // matlab header traits
  template <int size,bool is_signed,bool is_int>
  struct Matlab_header_traits;

  template <>
  struct Matlab_header_traits<1, true, true> // char
  { 
    static int const value_type = miINT8;
    static vsip::impl::uint8_type const class_type = mxINT8_CLASS; 
  };

  template <>
  struct Matlab_header_traits<1, false, true> // unsigned char
  { 
    static int const value_type = miUINT8;
    static vsip::impl::uint8_type const class_type = mxUINT8_CLASS; 
  };

  template <>
  struct Matlab_header_traits<2, true, true> // short
  { 
    static int const value_type = miINT16;
    static vsip::impl::uint8_type const class_type = mxINT16_CLASS; 
  };

  template <>
  struct Matlab_header_traits<2, false, true> // unsigned short
  { 
    static int const value_type = miUINT16;
    static vsip::impl::uint8_type const class_type = mxUINT16_CLASS; 
  };

  template <>
  struct Matlab_header_traits<4, true, true> // int
  { 
    static int const value_type= miINT32;
    static vsip::impl::uint8_type const class_type= mxINT32_CLASS;
  };

  template <>
  struct Matlab_header_traits<4, false, true> // unsigned int
  { 
    static int const value_type= miUINT32;
    static vsip::impl::uint8_type const class_type= mxUINT32_CLASS;
  };

  template <>
  struct Matlab_header_traits<4, true, false> // float
  { 
    static int const value_type= miSINGLE;
    static vsip::impl::uint8_type const class_type= mxSINGLE_CLASS;
  };

  template <>
  struct Matlab_header_traits<8, true, false> // double
  { 
    static int const value_type= miDOUBLE;
    static vsip::impl::uint8_type const class_type= mxDOUBLE_CLASS;
  };

  // matlab desired layouts
  template <template <typename,typename> class View>
  struct Matlab_desired_LP
  {
    static vsip::dimension_type const dim = vsip::impl::Dim_of_view<View>::dim;
    typedef vsip::impl::Layout<dim,
                     typename vsip::impl::Col_major<dim>::type,
                     vsip::impl::Stride_unit_dense,
		     vsip::impl::Cmplx_split_fmt> type;
  };



// Return Matlab data type as a string.

inline
char const*
data_type(int dt)
{
  switch (dt)
  {
  case miINT8:       return "miINT8";
  case miUINT8:      return "miUINT8";
  case miINT16:      return "miINT16";
  case miUINT16:     return "miUINT16";
  case miINT32:      return "miINT32";
  case miUINT32:     return "miUINT32";
  case miSINGLE:     return "miSINGLE";
  case miDOUBLE:     return "miDOUBLE";
  case miINT64:      return "miINT64";
  case miUINT64:     return "miUINT64";
  case miMATRIX:     return "miMATRIX";
  case miCOMPRESSED: return "miCOMPRESSED";
  case miUTF8:       return "miUTF8";
  case miUTF16:      return "miUTF16";
  case miUTF32:      return "miUTF32";
  }
  return "*unknown*";
}



// Return Matlab class type as a string.

inline
char const*
class_type(int ct)
{
  switch(ct)
  {
  case mxCELL_CLASS:   return "mxCELL_CLASS";
  case mxSTRUCT_CLASS: return "mxSTRUCT_CLASS";
  case mxOBJECT_CLASS: return "mxOBJECT_CLASS";
  case mxCHAR_CLASS:   return "mxCHAR_CLASS";
  case mxSPARSE_CLASS: return "mxSPARSE_CLASS";
  case mxDOUBLE_CLASS: return "mxDOUBLE_CLASS";
  case mxSINGLE_CLASS: return "mxSINGLE_CLASS";
  case mxINT8_CLASS:   return "mxINT8_CLASS";
  case mxUINT8_CLASS:  return "mxUINT8_CLASS";
  case mxINT16_CLASS:  return "mxINT16_CLASS";
  case mxUINT16_CLASS: return "mxUINT16_CLASS";
  case mxINT32_CLASS:  return "mxINT32_CLASS";
  case mxUINT32_CLASS: return "mxUINT32_CLASS";
  }
  return "*unknown*";
}


// Return whether Matlab class type is integer.

inline
bool
class_type_is_int(int ct)
{
  switch (ct)
  {
  case mxINT8_CLASS:
  case mxUINT8_CLASS:
  case mxINT16_CLASS:
  case mxUINT16_CLASS:
  case mxINT32_CLASS:
  case mxUINT32_CLASS: return true;
  }
  return false;
}


// Return whether Matlab class type is signed.

inline
bool
class_type_is_signed(int ct)
{
  switch (ct)
  {
  case mxINT8_CLASS:
  case mxINT16_CLASS:
  case mxINT32_CLASS:
  case mxSINGLE_CLASS:
  case mxDOUBLE_CLASS: return true;
  }
  return false;
}

} // namespace matlab

} // namespace vsip_csl

#endif // VSIP_CSL_MATLAB_HPP
