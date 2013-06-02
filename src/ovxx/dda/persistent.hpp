//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dda_persistent_hpp_
#define ovxx_dda_persistent_hpp_

#include <vsip/dda.hpp>
#include <ovxx/layout.hpp>

namespace ovxx
{
namespace dda
{
// A variant of DDA that supports longer life-times.
// Instead of storing a block reference this data object
// uses block_traits<B>::ptr_type to make sure the block remains
// valid. (This technique might be too costly for ordinary dda::Data.)
template <typename B,
         typename L  = typename dda::dda_block_layout<B>::layout_type>
class Persistent_data
{
  typedef Accessor<B, L, dda::in> backend_type;

public:

  typedef typename B::value_type value_type;
  typedef L layout_type;

  typedef typename backend_type::non_const_ptr_type non_const_ptr_type;
  typedef typename backend_type::const_ptr_type const_ptr_type;
  typedef typename conditional<is_modifiable_block<B>::value,
			       non_const_ptr_type,
			       const_ptr_type>::type ptr_type;

  static int const ct_cost = backend_type::ct_cost;

  Persistent_data(B &block,
		  dda::sync_policy sync = dda::inout,
		  non_const_ptr_type buffer = non_const_ptr_type())
    : block_(&block),
      buffer_(buffer),
      sync_(sync),
      backend_(*block_, buffer_)
  {}

  ~Persistent_data() {}
  void sync_in() { if (sync_ & dda::in) backend_.sync_in();}
  void sync_out() { if (sync_ & dda::out) backend_.sync_out();}

  ptr_type ptr() { return backend_.ptr();}
  const_ptr_type ptr() const { return backend_.ptr();}
  non_const_ptr_type non_const_ptr()
  { return const_cast_<non_const_ptr_type>(backend_.ptr());}
  stride_type stride(dimension_type d) const { return backend_.stride(d);}
  length_type size(dimension_type d) const { return backend_.size(d);}
  length_type size() const { return backend_.size();}
  int cost() const { return backend_.cost();}

private:
  typename block_traits<B>::ptr_type block_;
  non_const_ptr_type buffer_;
  dda::sync_policy sync_;
  backend_type backend_;
};

} // namespace ovxx::dda
} // namespace ovxx

#endif
