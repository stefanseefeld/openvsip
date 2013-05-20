//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_transposed_hpp_
#define ovxx_expr_transposed_hpp_

#include <ovxx/support.hpp>
#include <ovxx/parallel/transpose_map_decl.hpp>

namespace ovxx
{
namespace expr
{

/// The Transposed class exchanges the order of indices to a
/// 2-dimensional block, and the dimensions visible via 2-argument
/// size().
template <typename Block>
class Transposed : ovxx::ct_assert<Block::dim == 2>,
                   ovxx::detail::nonassignable
{
  typedef parallel::transpose_map_of<2, typename Block::map_type> map_functor;

public:
  static dimension_type const dim = Block::dim;
  typedef typename Block::value_type value_type;
  typedef value_type&                reference_type;
  typedef value_type const&          const_reference_type;
  typedef typename map_functor::type map_type;

  Transposed(Block &block) VSIP_NOTHROW
  : block_(&block),
    map_(map_functor::project(block.map()))
  { map_.impl_apply(block_domain<dim>(*this));}

  Transposed(Transposed const& b)
    : block_(&*b.block_),        // &* work's around holder's lack of copy-cons.
      map_ (b.map_)
  { map_.impl_apply(block_domain<dim>(*this));}

    ~Transposed() VSIP_NOTHROW {}

  length_type size() const VSIP_NOTHROW
  { return block_->size();}
  length_type size(dimension_type block_d, dimension_type d) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION (block_d == 2);
    OVXX_PRECONDITION (d <= 1);
    return block_->size(block_d, !d);
  }
  // These are noops as Transposed_block is held by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return map_; }

  // Data accessors.
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  { return block_->get(j, i);}

  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  { return block_->put(j, i, val);}

  reference_type ref(index_type i, index_type j) VSIP_NOTHROW
  { return block_->ref(j, i);}

  typedef storage_traits<value_type, get_block_layout<Block>::storage_format> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;

  ptr_type ptr() VSIP_NOTHROW { return block_->ptr();}
  const_ptr_type ptr() const VSIP_NOTHROW { return block_->ptr();}

  stride_type stride(dimension_type Dim OVXX_UNUSED, dimension_type d)
     const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    return block_->stride(dim, 1 - d);
  }

  Block const &block() const { return *this->block_;}

 private:
  // Data members.
  typename block_traits<Block>::ptr_type block_;
  map_type map_;
};

} // namespace expr

template <typename B>
struct block_traits<expr::Transposed<B> >
  : by_value_traits<expr::Transposed<B> > {};

template <typename B>
struct is_modifiable_block<expr::Transposed<B> >
  : is_modifiable_block<B> {};

namespace detail
{

template <typename B>
expr::Transposed<typename distributed_local_block<B>::type>
get_local_block(expr::Transposed<B> const &block)
{
  typedef typename distributed_local_block<B>::type super_type;
  typedef expr::Transposed<super_type> local_block_type;

  typename block_traits<super_type>::plain_type
    super_block = get_local_block(block.block());

  return local_block_type(super_block);
}

template <typename B>
expr::Transposed<typename distributed_local_block<B>::proxy_type>
get_local_proxy(expr::Transposed<B> const &block,
		index_type sb)
{
  typedef typename distributed_local_block<B>::proxy_type super_type;
  typedef expr::Transposed<super_type> local_proxy_type;

  typename block_traits<super_type>::plain_type
    super_block = get_local_proxy(block.block(), sb);

  return local_proxy_type(super_block);
}

template <typename B>
void assert_local(expr::Transposed<B> const &block, index_type sb)
{
  assert_local(block.block(), sb);
}

} // namespace ovxx::detail

template <typename B>
struct distributed_local_block<expr::Transposed<B> >
{
  typedef expr::Transposed<typename distributed_local_block<B>::type> type;
  typedef expr::Transposed<typename distributed_local_block<B>::proxy_type> proxy_type;
};

template <typename B>
struct lvalue_factory_type<expr::Transposed<B> >
{
  typedef typename lvalue_factory_type<B>
    ::template rebind<expr::Transposed<B> >::type type;
  template <typename O>
  struct rebind 
  {
    typedef typename lvalue_factory_type<B>
      ::template rebind<O>::type type;
  };
};

} // namespace ovxx

namespace vsip
{
namespace impl
{
// Take transpose of dimension-order.
template <typename T>
struct Transpose_order;

template <dimension_type Dim0,
	  dimension_type Dim1,
	  dimension_type Dim2>
struct Transpose_order<tuple<Dim0, Dim1, Dim2> >
{
  typedef tuple<Dim1, Dim0, Dim2> type;
};

} // namespace vsip::impl

/// dimension-order is reversed, pack-type becomes unknown.
template <typename B>
struct get_block_layout<ovxx::expr::Transposed<B> >
{
  static dimension_type const dim = B::dim;

  typedef typename impl::Transpose_order<
                      typename get_block_layout<B>::order_type>::type
					            order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename Block>
struct supports_dda<ovxx::expr::Transposed<Block> >
{ static bool const value = supports_dda<Block>::value;};

} // namespace vsip

#endif
