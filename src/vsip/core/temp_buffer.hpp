//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_TEMP_BUFFER_HPP
#define VSIP_CORE_TEMP_BUFFER_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <utility>

#include <vsip/support.hpp>
#include <vsip/core/noncopyable.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

template <typename T>
class Temp_buffer : Non_copyable
{
public:
  Temp_buffer(std::ptrdiff_t size)
    VSIP_THROW((std::bad_alloc))
    : data_ (0),
      size_ (size)
  {
    is_alloc_ = false;
    std::pair<T*, std::ptrdiff_t> p = std::get_temporary_buffer<T>(size_);

    if (p.second == 0)
    {
      is_alloc_ = true;
      data_     = new T[size_];
    }
    else if (p.second < size_)
    {
      std::return_temporary_buffer(p.first);
      is_alloc_ = true;
      data_     = new T[size_];
    }
    else
    {
      data_ = p.first;
    }
  }

  ~Temp_buffer()
    VSIP_NOTHROW
  {
    if (is_alloc_) 
      delete[] data_;
    else
      std::return_temporary_buffer(data_);
  }

  T* ptr() const { return data_; }

private:
  T*		 data_;
  std::ptrdiff_t size_;
  bool           is_alloc_;
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_TEMP_BUFFER_HPP
