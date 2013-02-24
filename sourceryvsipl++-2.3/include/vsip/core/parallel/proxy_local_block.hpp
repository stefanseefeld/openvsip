/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/proxy_local_block.hpp
    @author  Jules Bergmann
    @date    2005-08-21
    @brief   VSIPL++ Library: ...

*/

#ifndef VSIP_CORE_PROXY_LOCAL_BLOCK_HPP
#define VSIP_CORE_PROXY_LOCAL_BLOCK_HPP

#include <vsip/core/layout.hpp>
#include <vsip/core/parallel/local_map.hpp>

namespace vsip
{
namespace impl
{

template <dimension_type Dim,
	  typename       T,
	  typename       LP>
class Proxy_local_block
{
  // Compile-time values and types.
public:
  static dimension_type const dim = Dim;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef Local_map map_type;

  // Implementation types.
public:
  typedef LP                       layout_type;
  typedef impl::Applied_layout<LP> applied_layout_type;

  // Constructors and destructor.
public:
  Proxy_local_block(Length<Dim> const& size)
    : layout_(size)
  {}

  // Data accessors.
public:
  // No get, put.

  // Direct_data interface.
public:
  stride_type impl_offset() VSIP_NOTHROW
  { return 0; }

  // NO // impl_data_type       impl_data()       VSIP_NOTHROW

  // NO // impl_const_data_type impl_data() const VSIP_NOTHROW

  stride_type impl_stride(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    assert((block_dim == 1 || block_dim == Dim) && (d < block_dim));
    return (block_dim == 1) ? 1 : layout_.stride(d);
  }

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW
  {
    length_type retval = layout_.size(0);
    for (dimension_type d=1; d<Dim; ++d)
      retval *= layout_.size(d);
    return retval;
  }

  length_type size(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    assert((block_dim == 1 || block_dim == Dim) && (d < block_dim));
    return (block_dim == 1) ? this->size() : this->layout_.size(d);
  }

  map_type const& map() const VSIP_NOTHROW { return map_; }


  // Member data.
private:
  applied_layout_type layout_;
  map_type            map_;
};



// Store Proxy_local_block by-value.
template <dimension_type Dim,
	  typename       T,
	  typename       LP>
struct View_block_storage<Proxy_local_block<Dim, T, LP> >
  : By_value_block_storage<Proxy_local_block<Dim, T, LP> >
{};


template <dimension_type Dim,
	  typename       T,
	  typename       LP>
struct Block_layout<Proxy_local_block<Dim, T, LP> >
{
  static dimension_type const dim = Dim;

  typedef Direct_access_tag         access_type;
  typedef typename LP::order_type   order_type;
  typedef typename LP::pack_type    pack_type;
  typedef typename LP::complex_type complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> layout_type;
};

} // namespace vsip::impl
} // namespace vsip



#endif // VSIP_CORE_PROXY_LOCAL_BLOCK_HPP
