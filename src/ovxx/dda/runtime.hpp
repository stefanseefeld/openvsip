//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_dda_runtime_hpp_
#define ovxx_dda_runtime_hpp_

#include <vsip/dda.hpp>
#include <ovxx/layout.hpp>

namespace vsip
{
namespace dda
{
/// Specialization for modifiable blocks, using a runtime layout
template <typename B, sync_policy S, dimension_type D, bool A>
class Data<B, S, ovxx::Rt_layout<D>, false, A> : ovxx::detail::noncopyable
{
  typedef B block_type;
  typedef ovxx::dda::Accessor<B, ovxx::Rt_layout<D>, S> backend_type;

public:
  typedef typename B::value_type value_type;
  typedef ovxx::Rt_layout<B::dim> layout_type;
  typedef typename backend_type::non_const_ptr_type ptr_type;
  typedef typename backend_type::const_ptr_type const_ptr_type;

  static int const ct_cost = backend_type::ct_cost;

  Data(block_type &block,
       layout_type const &rtl,
       ptr_type buffer = ptr_type(),
       length_type buffer_size = 0)
    : backend_(block, rtl, false, buffer, buffer_size)
  {}

  ~Data() {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  ptr_type ptr() { return backend_.ptr();}
  const_ptr_type ptr() const { return backend_.ptr();}
  stride_type  stride(dimension_type d) const { return backend_.stride(d);}
  length_type  size(dimension_type d) const { return backend_.size(d);}
  length_type  size() const { return backend_.size();}
  int cost() const { return backend_.cost();}

private:
  backend_type backend_;
};

/// Specialization for unmodifiable data with runtime layout
template <typename B, sync_policy S, dimension_type D>
class Data<B, S, ovxx::Rt_layout<D>, true, false> : ovxx::detail::noncopyable
{
  typedef B block_type;
  typedef ovxx::dda::Accessor<B const, ovxx::Rt_layout<D>, S> backend_type;

public:
  typedef typename B::value_type value_type;
  typedef ovxx::Rt_layout<B::dim> layout_type;
  typedef typename backend_type::non_const_ptr_type non_const_ptr_type;
  typedef typename backend_type::const_ptr_type ptr_type;

  static int const ct_cost = backend_type::ct_cost;

  Data(block_type const &block,
       layout_type const &rtl,
       non_const_ptr_type buffer = non_const_ptr_type(),
       length_type buffer_size = 0)
    : backend_(block, rtl, false, buffer, buffer_size)
  {}

  Data(block_type const &block,
       bool force_copy,
       layout_type const &rtl,
       non_const_ptr_type buffer = non_const_ptr_type(),
       length_type buffer_size = 0)
    : backend_(block, rtl, force_copy, buffer, buffer_size)
  {}

  ~Data() {}

  void sync_in() { backend_.sync_in();}
  void sync_out() { backend_.sync_out();}

  ptr_type ptr() const { return backend_.ptr();}
  non_const_ptr_type non_const_ptr() { return ovxx::const_cast_<non_const_ptr_type>(ptr());}
  stride_type  stride(dimension_type d) const { return backend_.stride(d);}
  length_type  size(dimension_type d) const { return backend_.size(d);}
  length_type  size() const { return backend_.size();}
  int cost() const { return backend_.cost();}

private:
  backend_type backend_;
};

} // namespace vsip::dda
} // namespace vsip

namespace ovxx
{
namespace dda
{
template <typename B, dda::sync_policy S, bool ReadOnly = !(S&dda::out)>
class Rt_data : public vsip::dda::Data<B, S, ovxx::Rt_layout<B::dim> >
{
  typedef vsip::dda::Data<B, S, ovxx::Rt_layout<B::dim> > base_type;
  typedef ovxx::Rt_layout<B::dim> layout_type;

public:
  typedef typename base_type::ptr_type ptr_type;

  Rt_data(B &block,
	  layout_type const &rtl,
	  ptr_type buffer = ptr_type(),
	  length_type size = 0)
    : base_type(block, rtl, buffer, size) {}
};

template <typename B, dda::sync_policy S>
class Rt_data<B, S, true> : public vsip::dda::Data<B, S, ovxx::Rt_layout<B::dim> >
{
  typedef vsip::dda::Data<B, S, ovxx::Rt_layout<B::dim> > base_type;
  typedef ovxx::Rt_layout<B::dim> layout_type;

public:
  typedef typename base_type::non_const_ptr_type non_const_ptr_type;

  Rt_data(B const &block,
	  layout_type const &rtl,
	  non_const_ptr_type buffer = non_const_ptr_type(),
	  length_type size = 0)
    : base_type(block, rtl, buffer, size) {}

  Rt_data(B const &block,
	  bool force_copy,
	  layout_type const &rtl,
	  non_const_ptr_type buffer = non_const_ptr_type(),
	  length_type size = 0)
    : base_type(block, force_copy, rtl, buffer, size) {}
};

} // namespace ovxx::dda
} // namespace ovxx

#endif
