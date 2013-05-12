//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_proxy_local_block_hpp_
#define ovxx_parallel_proxy_local_block_hpp_

#include <ovxx/layout.hpp>
#include <vsip/impl/local_map.hpp>

namespace ovxx
{
namespace parallel
{

template <dimension_type D, typename T, typename L>
class proxy_local_block
{
public:
  static dimension_type const dim = D;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef Local_map map_type;

  typedef L layout_type;
  typedef Applied_layout<L> applied_layout_type;

  proxy_local_block(Length<D> const &size) : layout_(size) {}

  // No get, put.

  stride_type offset() VSIP_NOTHROW { return 0;}
  stride_type stride(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION((block_dim == 1 || block_dim == D) && (d < block_dim));
    return (block_dim == 1) ? 1 : layout_.stride(d);
  }

  length_type size() const VSIP_NOTHROW
  {
    length_type retval = layout_.size(0);
    for (dimension_type d=1; d<D; ++d)
      retval *= layout_.size(d);
    return retval;
  }

  length_type size(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION((block_dim == 1 || block_dim == D) && (d < block_dim));
    return (block_dim == 1) ? this->size() : this->layout_.size(d);
  }

  map_type const &map() const VSIP_NOTHROW { return map_;}

private:
  applied_layout_type layout_;
  map_type            map_;
};

} // namespace ovxx::parallel

template <dimension_type D, typename T, typename L>
struct block_traits<parallel::proxy_local_block<D, T, L> >
  : by_value_traits<parallel::proxy_local_block<D, T, L> >
{};

} // namespace ovxx

namespace vsip
{
template <dimension_type D, typename T, typename L>
struct get_block_layout<ovxx::parallel::proxy_local_block<D, T, L> >
{
  static dimension_type const dim = D;
  typedef typename L::order_type   order_type;
  static pack_type const packing = L::packing;
  static storage_format_type const storage_format = L::storage_format;
  
  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <dimension_type D, typename T, typename L>
struct supports_dda<ovxx::parallel::proxy_local_block<D, T, L> >
{ static bool const value = true;};

} // namespace vsip

#endif
