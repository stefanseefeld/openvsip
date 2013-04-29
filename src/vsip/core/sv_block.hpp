//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_SV_BLOCK_HPP
#define VSIP_CORE_SV_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>

#include <vsip/core/refcount.hpp>
#include <vsip/core/block_traits.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{



template <typename T>
class Sv_local_block
  : public impl::Ref_count<Sv_local_block<T> >
{
  // Compile-time values and types.
  typedef std::vector<T> vector_type;
public:
  static dimension_type const dim = 1;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef row1_type order_type;
  typedef Local_map map_type;

  // Implementation types.
public:

  // Constructors and destructor.
public:
  Sv_local_block(Domain<1> const& dom, map_type const& = map_type())
    VSIP_THROW((std::bad_alloc))
    : vector_(dom[0].size())
  {}

  Sv_local_block(Domain<1> const& dom, T value, map_type const& = map_type())
    VSIP_THROW((std::bad_alloc))
    : vector_(dom[0].size())
  {
    for (index_type i=0; i<dom[0].size(); ++i)
      this->put(i, value);
  }

  ~Sv_local_block() VSIP_NOTHROW
  {}

  // Data accessors.
public:
  T get(index_type idx) const VSIP_NOTHROW
  {
    assert(idx < size());
    return vector_[idx];
  }

  void put(index_type idx, T val) VSIP_NOTHROW
  {
    assert(idx < size());
    vector_[idx] = val;
  }

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW
    { return vector_.size(); }
  length_type size(dimension_type D, dimension_type d) const VSIP_NOTHROW
    { assert(D == 1 && d == 0); return vector_.size(); }

  map_type const& map() const VSIP_NOTHROW { return map_; }

  vector_type const& impl_vector() { return vector_; }

  // Hidden copy constructor and assignment.
private:
  Sv_local_block(Sv_local_block const&);
  Sv_local_block& operator=(Sv_local_block const&);

  // Member Data
private:
  vector_type vector_;
  map_type    map_;
};


} // namespace impl
} // namespace vsip

#endif // VSIP_CORE_SV_BLOCK_HPP
