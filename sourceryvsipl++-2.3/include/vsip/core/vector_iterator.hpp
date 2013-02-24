/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/vector-iterator.hpp
    @author  Jules Bergmann
    @date    2006-01-04
    @brief   VSIPL++ Library: Vector Iterator.

    Iterator over a sequence of values stored in a vector view.
*/

#ifndef VSIP_CORE_VECTOR_ITERATOR_HPP
#define VSIP_CORE_VECTOR_ITERATOR_HPP

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


/// Class to iterate over values stored in a vector view.

template <typename ViewT>
class Vector_iterator
{
public:
  typedef typename ViewT::value_type value_type;

  Vector_iterator(ViewT view, index_type idx)
    : view_(view),
      idx_ (idx)
    {}

  Vector_iterator(Vector_iterator const& rhs)
    : view_(rhs.view_),
      idx_ (rhs.idx_)
    {}

  Vector_iterator& operator=(Vector_iterator const& rhs)
  {
    view_  = rhs.view_;
    idx_   = rhs.idx_;
    return *this;
  }

  Vector_iterator& operator++()       { idx_ += 1;  return *this; }
  Vector_iterator& operator--()       { idx_ -= 1;  return *this; }
  Vector_iterator& operator+=(int dx) { idx_ += dx; return *this; }
  Vector_iterator& operator-=(int dx) { idx_ -= dx; return *this; }

  Vector_iterator& operator++(int)
    { Vector_iterator tmp = *this; idx_ += 1; return tmp; }
  Vector_iterator& operator--(int)
    { Vector_iterator tmp = *this; idx_ -= 1; return tmp; }

  bool operator==(Vector_iterator const& rhs) const
    { return &(view_.block()) == &(view_.block()) && idx_ == rhs.idx_; }

  bool operator!=(Vector_iterator const& rhs) const
    { return &(view_.block()) != &(view_.block()) || idx_ != rhs.idx_; }

  bool operator<(Vector_iterator const& rhs) const
    { return (idx_ < rhs.idx_); }

  int operator-(Vector_iterator const& rhs) const
    { return (idx_ - rhs.idx_); }

  Vector_iterator operator+(int dx) const
  {
    Vector_iterator res(view_, idx_);
    res += dx;
    return res;
  }
  Vector_iterator operator-(int dx) const
  {
    Vector_iterator res(view_, idx_);
    res -= dx;
    return res;
  }
  
  value_type operator*() const
    { return view_.get(idx_); }

  // Member data.
private:
  ViewT         view_;
  index_type	idx_;
};


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_VECTOR_ITERATOR_HPP
