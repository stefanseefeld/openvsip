//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_VALUE_ITERATOR_HPP
#define VSIP_CORE_VALUE_ITERATOR_HPP

/***********************************************************************
  Included Files
***********************************************************************/



/***********************************************************************
  Declarations & Class Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{


/// Class to iterate over a sequence of values.

template <typename T,
	  typename DeltaT>
class Value_iterator
{
public:
  typedef T      value_type;
  typedef DeltaT delta_type;

  Value_iterator(value_type val, delta_type delta)
    : current_(val),
      delta_  (delta)
    {}

  Value_iterator(Value_iterator const& rhs)
    : current_(rhs.current_),
      delta_  (rhs.delta_)
    {}

  Value_iterator& operator=(Value_iterator const& rhs)
  {
    current_ = rhs.current_;
    delta_   = rhs.delta_;
    return *this;
  }

  Value_iterator& operator++()       { current_ += delta_; return *this; }
  Value_iterator& operator--()       { current_ -= delta_; return *this; }
  Value_iterator& operator+=(int dx) { current_ += dx * delta_; return *this; }
  Value_iterator& operator-=(int dx) { current_ -= dx * delta_; return *this; }

  Value_iterator& operator++(int)
    { Value_iterator tmp = *this; current_ += delta_; return tmp; }
  Value_iterator& operator--(int)
    { Value_iterator tmp = *this; current_ -= delta_; return tmp; }

  bool operator==(Value_iterator const& rhs) const
    { return (current_ == rhs.current_ && delta_ == rhs.delta_); }

  bool operator!=(Value_iterator const& rhs) const
    { return (current_ != rhs.current_ || delta_ != rhs.delta_); }

  bool operator<(Value_iterator const& rhs) const
    { return current_ < rhs.current_; }

  int operator-(Value_iterator const& rhs) const
    { return (current_ - rhs.current_) / delta_; }

  Value_iterator operator+(int dx) const
  {
    Value_iterator res(current_, delta_);
    res += dx;
    return res;
  }
  Value_iterator operator-(int dx) const
  {
    Value_iterator res(current_, delta_);
    res -= dx;
    return res;
  }
  
  value_type operator*() const
    { return current_; }

  // Member data.
private:
  value_type	current_;
  delta_type	delta_;
};


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_VALUE_ITERATOR_HPP
