//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_value_iterator_hpp_
#define ovxx_value_iterator_hpp_

namespace ovxx
{

/// Class to iterate over a sequence of equally-spaced values.
template <typename T, typename D>
class value_iterator
{
public:
  typedef T value_type;
  typedef D delta_type;

  value_iterator(value_type val, delta_type delta)
    : current_(val),
      delta_(delta)
  {}

  value_iterator(value_iterator const &other)
    : current_(other.current_),
      delta_(other.delta_)
  {}

  value_iterator &operator=(value_iterator const &other)
  {
    current_ = other.current_;
    delta_   = other.delta_;
    return *this;
  }

  value_iterator &operator++()       { current_ += delta_; return *this;}
  value_iterator &operator--()       { current_ -= delta_; return *this;}
  value_iterator &operator+=(int dx) { current_ += dx * delta_; return *this;}
  value_iterator &operator-=(int dx) { current_ -= dx * delta_; return *this;}

  value_iterator &operator++(int)
  { value_iterator tmp = *this; current_ += delta_; return tmp;}
  value_iterator &operator--(int)
  { value_iterator tmp = *this; current_ -= delta_; return tmp;}

  bool operator==(value_iterator const &other) const
  { return (current_ == other.current_ && delta_ == other.delta_);}

  bool operator!=(value_iterator const &other) const
  { return (current_ != other.current_ || delta_ != other.delta_);}

  bool operator<(value_iterator const &other) const
  { return current_ < other.current_;}

  int operator-(value_iterator const &other) const
  { return (current_ - other.current_) / delta_;}

  value_iterator operator+(int dx) const
  {
    value_iterator res(current_, delta_);
    res += dx;
    return res;
  }
  value_iterator operator-(int dx) const
  {
    value_iterator res(current_, delta_);
    res -= dx;
    return res;
  }
  
  value_type operator*() const { return current_;}

private:
  value_type current_;
  delta_type delta_;
};

} // namespace ovxx

#endif
