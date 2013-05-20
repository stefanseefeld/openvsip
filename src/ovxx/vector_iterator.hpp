//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_vector_iterator_hpp_
#define ovxx_vector_iterator_hpp_

namespace ovxx
{

/// Class to iterate over values stored in a vector
template <typename V>
class vector_iterator
{
public:
  typedef typename V::value_type value_type;

  vector_iterator(V view, index_type idx)
    : view_(view),
      idx_ (idx)
    {}

  vector_iterator &operator++() { idx_ += 1; return *this;}
  vector_iterator &operator--() { idx_ -= 1; return *this;}
  vector_iterator &operator+=(int dx) { idx_ += dx; return *this;}
  vector_iterator &operator-=(int dx) { idx_ -= dx; return *this;}
  vector_iterator& operator++(int)
  { vector_iterator tmp = *this; idx_ += 1; return tmp;}
  vector_iterator& operator--(int)
  { vector_iterator tmp = *this; idx_ -= 1; return tmp;}

  bool operator==(vector_iterator const &rhs) const
  { return &(view_.block()) == &(view_.block()) && idx_ == rhs.idx_;}

  bool operator!=(vector_iterator const &rhs) const
  { return &(view_.block()) != &(view_.block()) || idx_ != rhs.idx_;}

  bool operator<(vector_iterator const& rhs) const
  { return (idx_ < rhs.idx_);}

  int operator-(vector_iterator const& rhs) const
  { return (idx_ - rhs.idx_);}

  vector_iterator operator+(int dx) const
  {
    vector_iterator res(view_, idx_);
    res += dx;
    return res;
  }
  vector_iterator operator-(int dx) const
  {
    vector_iterator res(view_, idx_);
    res -= dx;
    return res;
  }
  
  value_type operator*() const
  { return view_.get(idx_);}

private:
  V view_;
  index_type idx_;
};

} // namespace ovxx

#endif
