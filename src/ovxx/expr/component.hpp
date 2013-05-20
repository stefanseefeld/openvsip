//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_component_hpp_
#define ovxx_expr_component_hpp_

#include <ovxx/support.hpp>

namespace ovxx
{
namespace expr
{
namespace op
{
template <typename C>
struct RealC
{
  typedef typename C::value_type value_type;

  static value_type get(C val) VSIP_NOTHROW { return val.real();}
  static void set(C &elt, value_type val) VSIP_NOTHROW 
  { elt = C(val, elt.imag());}
  
  static value_type *get_ptr(C *data)
  { return reinterpret_cast<value_type*>(data);}

  static value_type *get_ptr(std::pair<value_type*, value_type*> const& data)
  { return data.first;}
};

template <typename C>
struct ImagC
{
  typedef typename C::value_type  value_type;

  static value_type get(C val) VSIP_NOTHROW { return val.imag();}
  static void set(C &elt, value_type val) VSIP_NOTHROW 
  { elt = C(elt.real(), val);}

  static value_type *get_ptr(C *data)
  { return reinterpret_cast<value_type*>(data) + 1;}

  static value_type *get_ptr(std::pair<value_type*, value_type*> const& data)
  { return data.second;}
};

} // namespace ovxx::expr::op

/// The Component_block class applies an "Extractor" policy to all
/// accesses via get() and put().  Everything else is deferred to the
/// underlying block.
template <typename Block, template <typename> class Extractor>
class Component : ovxx::detail::nonassignable
{
  typedef Extractor<typename Block::value_type> extr_type;
public:
  // Compile-time values and types.
  // The type of elements of this block is determined by the Extractor
  // instantiated for the Block's value_type.
  static dimension_type const dim = Block::dim;
  typedef typename extr_type::value_type  value_type;
  typedef value_type&       reference_type;
  typedef value_type const& const_reference_type;
  typedef typename Block::map_type map_type;

  Component(Component const &b) : block_(&*b.block_) {}
  Component(Block &block) VSIP_NOTHROW : block_ (&block) {}
  ~Component() VSIP_NOTHROW {}

  // Accessors.
  length_type size() const VSIP_NOTHROW { return block_->size();}
  length_type size(dimension_type block_d, dimension_type d) const VSIP_NOTHROW
  { return block_->size(block_d, d);}
  // These are noops as Component_block is helt by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return block_->map();}

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW
  { return extr_type::get(block_->get(i));}
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  { return extr_type::get(block_->get(i, j));}
  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  { return extr_type::get(block_->get(i, j, k));}

  void put(index_type i, value_type val) VSIP_NOTHROW
  {
    typename Block::value_type tmp = block_->get(i);
    extr_type::set(tmp, val);
    block_->put(i, tmp);
  }
  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  {
    typename Block::value_type tmp = block_->get(i, j);
    extr_type::set(tmp, val);
    block_->put(i, j, tmp);
  }
  void put(index_type i, index_type j, index_type k, value_type val) VSIP_NOTHROW
  {
    typename Block::value_type tmp = block_->get(i, j, k);
    extr_type::set(tmp, val);
    block_->put(i, j, k, tmp);
  }

  typedef storage_traits<value_type, array> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;

  ptr_type ptr() VSIP_NOTHROW
  { 
    return extr_type::get_ptr(block_->ptr());
  }

  const_ptr_type ptr() const VSIP_NOTHROW
  { 
    return extr_type::get_ptr(block_->ptr());
  }

  stride_type stride(dimension_type Dim OVXX_UNUSED, dimension_type d)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    if (get_block_layout<Block>::storage_format == split_complex)
      return 1*block_->stride(dim, d);
    else
      return 2*block_->stride(dim, d);
  }

  Block const &block() const { return *this->block_;}
  Block &block() { return *this->block_;}

private:
  // Data members.
  typename block_traits<Block>::ptr_type block_;
};

} // namespace ovxx::expr

template <typename B, template <typename> class E>
struct block_traits<expr::Component<B, E> >
  : by_value_traits<expr::Component<B, E> > {};

template <typename B, template <typename> class E>
struct is_modifiable_block<expr::Component<B, E> >
  : is_modifiable_block<B> {};

template <typename Block,
          template <typename> class Extractor>
struct distributed_local_block<expr::Component<Block, Extractor> >
{
  typedef expr::Component<typename distributed_local_block<Block>::type,
			  Extractor> type;
  typedef expr::Component<typename distributed_local_block<Block>::proxy_type,
			  Extractor> proxy_type;
};

namespace detail
{

template <typename Block, template <typename> class Extractor>
expr::Component<typename distributed_local_block<Block>::type, Extractor>
get_local_block(expr::Component<Block, Extractor> const& block)
{
  typedef typename distributed_local_block<Block>::type super_type;
  typedef expr::Component<super_type, Extractor>        local_block_type;

  typename block_traits<super_type>::plain_type
    super_block = get_local_block(block.block());

  return local_block_type(super_block);
}

template <typename Block,
          template <typename> class Extractor>
expr::Component<typename distributed_local_block<Block>::proxy_type, Extractor>
get_local_proxy(expr::Component<Block, Extractor> const& block,
		index_type sb)
{
  typedef typename distributed_local_block<Block>::proxy_type super_type;
  typedef expr::Component<super_type, Extractor> local_proxy_type;

  typename block_traits<super_type>::plain_type
    super_block = get_local_proxy(block.block(), sb);

  return local_proxy_type(super_block);
}

} // namespace ovxx::detail
} // namespace ovxx

namespace vsip
{
/// If `Block` is split-complex, packing remains the same.
/// Otherwise it becomes unknown.
template <typename Block, template <typename> class Extractor>
struct get_block_layout<ovxx::expr::Component<Block, Extractor> >
{
  static dimension_type const dim = Block::dim;

  typedef typename get_block_layout<Block>::order_type   order_type;
  static storage_format_type const storage_format = array;

  static pack_type const packing = 
    get_block_layout<Block>::storage_format == split_complex ?
    get_block_layout<Block>::packing :
    any_packing;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename Block, template <typename> class Extractor>
struct supports_dda<ovxx::expr::Component<Block, Extractor> >
{ static bool const value = supports_dda<Block>::value;};

} // namespace vsip

#endif
