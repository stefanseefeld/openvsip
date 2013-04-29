/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_csl_matlab_file_hpp_
#define vsip_csl_matlab_file_hpp_

#include <iostream>
#include <fstream>
#include <vsip/core/noncopyable.hpp>
#include <vsip_csl/matlab_bin_formatter.hpp>

namespace vsip_csl
{

class Matlab_file : vsip::impl::Non_copyable
{
public:
  class iterator
  {
    friend class Matlab_file;
  public:
    iterator(iterator const &obj)
      : mf_(obj.mf_), end_iterator_(obj.end_iterator_) {}

    iterator&
    operator=(iterator const &src)
    {
      this->mf_           = src.mf_;
      this->end_iterator_ = src.end_iterator_;
      return *this;
    }

    void read_header() { mf_->is_ >> mf_->view_header_;}

    iterator& operator++()
    {
      if(!mf_->read_data_)
      {
	// advance file pointer to next header
	// make sure that we don't go beyond the end of file!
	if(mf_->view_header_.next_header >= mf_->length_)
	  mf_->end_of_file_ = true;
	else 
	  mf_->is_.seekg(mf_->view_header_.next_header);
	
      }
      if(!mf_->end_of_file_) // read next header
	read_header();
      
      mf_->read_data_ = false;
      return *this;
    }

    bool operator==(iterator &i1) { return mf_->end_of_file_ == i1.end_iterator_;}
    bool operator!=(iterator &i1) { return mf_->end_of_file_ != i1.end_iterator_;}
    Matlab_view_header *operator*() { return &(mf_->view_header_);}

  private:
    iterator(bool end_iterator,Matlab_file *mf)
      : mf_(mf), end_iterator_(end_iterator) {}
        
    Matlab_file *mf_;
    bool end_iterator_;
  };
    
  friend class iterator;

  Matlab_file(std::string fname);

  iterator begin() { return begin_iterator_; };
  iterator end() { return end_iterator_; };

  // read a view from a matlab file after reading the header
  template <typename T,
	    typename Block0,
	    template <typename,typename> class View>
  void read_view(View<T,Block0> view, iterator  &iter);

  Matlab_bin_hdr header() const { return matlab_header_;}

private:
  Matlab_bin_hdr                    matlab_header_;
  std::ifstream                     is_;
  iterator                          begin_iterator_;
  iterator                          end_iterator_;

  Matlab_view_header view_header_;
  bool read_data_;
  bool end_of_file_;
  vsip::impl::uint32_type length_;
};

Matlab_file::Matlab_file(std::string fname)
  : is_(fname.c_str()),
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

// read a view from a matlab file after reading the header
template <typename T,
	  typename Block0,
	  template <typename,typename> class View>
void Matlab_file::read_view(View<T,Block0> view, iterator  &iter)
{
  typedef vsip::impl::int8_type int8_type;
  typedef vsip::impl::uint8_type uint8_type;
  typedef vsip::impl::int16_type int16_type;
  typedef vsip::impl::uint16_type uint16_type;
  typedef vsip::impl::int32_type int32_type;
  typedef vsip::impl::uint32_type uint32_type;
  typedef vsip::impl::uint64_type uint64_type;

  typedef typename vsip::impl::scalar_of<T>::type scalar_type;
  vsip::dimension_type v_dim = vsip::impl::Dim_of_view<View>::dim;
  Matlab_view_header *header = *iter;
  std::istream *is = &(this->is_);

  // make sure that this iterator points to the same Matlab_file pointer
  assert(iter.mf_ == this);

  // make sure that both the view and the file are both complex or real
  if(vsip::impl::is_complex<T>::value && !header->is_complex)
    VSIP_IMPL_THROW(std::runtime_error(
      "Trying to read complex view into a real view"));

  // make sure that both the view and the file have the same class
  if(!(header->class_type == 
            (matlab::Matlab_header_traits<sizeof(scalar_type),
                  std::numeric_limits<scalar_type>::is_signed,
                  std::numeric_limits<scalar_type>::is_integer>::class_type)
	    ))
    VSIP_IMPL_THROW(std::runtime_error(
      "Trying to read a view of a different class"));

  // make sure that both the view and the file have the same dimensions
  if(v_dim == 1) header->num_dims--; // special case for vectors
  if(v_dim != header->num_dims)
    VSIP_IMPL_THROW(std::runtime_error(
      "Trying to read a view of different dimensions"));

  if(v_dim == 1)  // special case for vectors because they can be 1xN or Nx1
  {
    if( (view.size(0) != header->dims[0] && header->dims[1] == 1) ||
        (view.size(0) != header->dims[1] && header->dims[0] == 1) )
      VSIP_IMPL_THROW(std::runtime_error(
        "View dimensions don't agree"));
  }
  else
  {
    for(vsip::dimension_type i=0;i<v_dim;i++)
      if(view.size(i) != header->dims[i])
        VSIP_IMPL_THROW(std::runtime_error(
          "View dimensions don't agree"));
  }

  // read data, we will go in this loop twice if we have complex data
  for (int i=0;i <= vsip::impl::is_complex<T>::value;i++)
  {
    typedef matlab::Subview_helper<View<T,Block0> > subview;
    typedef typename subview::realview_type r_v;
    typedef typename subview::imagview_type i_v;
    bool     swap_value = header->swap_bytes;
    matlab::data_element temp_data_element;

    // read data header
    is->read(reinterpret_cast<char*>(&temp_data_element),
            sizeof(temp_data_element));

    // should we swap this field?
    bswap(temp_data_element.type,swap_value);
    bswap(temp_data_element.size,swap_value);


    // Because we don't know how the data was stored, we need to instantiate
    // generic_reader which can read a type and cast into a different one
    if(temp_data_element.type == matlab::miINT8) 
    {
      if(i==0)
        matlab::read<int8_type, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<int8_type, i_v>(*is, subview::imag(view), swap_value);
    }
    else if(temp_data_element.type == matlab::miUINT8) 
    {
      if(i==0)
        matlab::read<uint8_type, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<uint8_type, i_v>(*is, subview::imag(view), swap_value);
    }
    else if(temp_data_element.type == matlab::miINT16) 
    {
      if(i==0)
        matlab::read<int16_type, r_v>(*is, subview::real(view), swap_value);
      else   
        matlab::read<int16_type, i_v>(*is, subview::imag(view), swap_value);
    }
    else if(temp_data_element.type == matlab::miUINT16) 
      {
      if(i==0)
        matlab::read<uint16_type, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<uint16_type, i_v>(*is, subview::imag(view), swap_value);
    }
    else if(temp_data_element.type == matlab::miINT32) 
    {
      if(i==0)
        matlab::read<int32_type, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<int32_type, i_v>(*is, subview::imag(view), swap_value);
    }
    else if(temp_data_element.type == matlab::miUINT32) 
    {
      if(i==0)
        matlab::read<uint32_type, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<uint32_type, i_v>(*is, subview::imag(view), swap_value);
    }
    else if(temp_data_element.type == matlab::miSINGLE) 
    {
      if(i==0)
        matlab::read<float, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<float, i_v>(*is, subview::imag(view), swap_value);
    }
    else
    {
      if(i==0)
        matlab::read<double, r_v>(*is, subview::real(view), swap_value);
      else
        matlab::read<double, i_v>(*is, subview::imag(view), swap_value);
    }
  }
}
} // namespace vsip_csl

#endif
