//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_subset_hpp_
#define ovxx_expr_subset_hpp_

#include <ovxx/parallel/subset_map_decl.hpp>
#include <ovxx/parallel/service.hpp>

namespace ovxx
{
namespace expr
{
namespace detail
{

/// Functor to create map for a subset block from its parent block's map.
/// (not a map class)
template <dimension_type D, typename M>
struct subset_map
{
  typedef typename parallel::map_subdomain<D, M>::type type;

  static type convert_map(M const &map, Domain<D> const &dom)
  {
    return parallel::map_subdomain<D, M>::project(map, dom);
  }

  static index_type parent_subblock(M const &map, Domain<D> const &dom, index_type sb)
  {
    return parallel::map_subdomain<D, M>::parent_subblock(map, dom, sb);
  }
};

/// Helper class to translate a distributed subset into a local subset.
/// In general case, ask the parent block what the local subset should be.
template <typename B, typename M = typename B::map_type>
struct subset_parent_local_domain
{
  static dimension_type const dim = B::dim;

  static Domain<dim> parent_local_domain(B const &block, index_type sb)
  {
    return block.block().map().impl_local_from_global_domain(sb,
							     block.domain());
  }
};

/// Specialize for subset_map, where the local subset is currently larger than
/// what the parent thinks it should be.
template <typename B, dimension_type D>
struct subset_parent_local_domain<B, parallel::subset_map<D> >
{
  static dimension_type const dim = B::dim;

  static Domain<dim> parent_local_domain(B const& block, index_type sb)
  {
#if 0
    // consider asking subset_map about parent's local block,
    // rather than asking the parent to recompute it.
    return block.map().template impl_parent_local_domain<dim>(sb);
#else
    index_type parent_sb = block.map().impl_parent_subblock(sb);
    return block.block().map().impl_local_from_global_domain(parent_sb,
							     block.domain());
#endif
  }
};

} // namespace ovxx::expr::detail

/// The Subset class maps all accesses through a Domain instance
/// before forwarding to the underlying block.  Thus, for instance, if
/// a 1-dimensional vector of length N is the underlying block, and the
/// Domain instance in use is Domain<1>(N/2)*2, then the Subset class
/// will expose every other element of the underlying block.
template <typename B>
class Subset : ovxx::detail::nonassignable
{
  typedef storage_traits<typename B::value_type, get_block_layout<B>::storage_format>
    storage;
public:
  static dimension_type const dim = B::dim;
  typedef typename B::value_type value_type;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;

  typedef detail::subset_map<dim, typename B::map_type> map_functor;
  typedef typename map_functor::type map_type;

  Subset(Domain<dim> const& dom, B &block) VSIP_NOTHROW
    : block_ (&block),
      dom_ (dom),
      map_ (map_functor::convert_map(block.map(), dom_))
  {
    // Sanity check that all of the Domain indices are within the
    // underlying block's range.  (If domain is empty, value
    // returned by impl_last() is not valid.)
    for (dimension_type d = 0; d < dim; d++)
    {
      OVXX_PRECONDITION(dom_[d].size() == 0 || dom_[d].first() < block_->size(dim, d));
      OVXX_PRECONDITION(dom_[d].size() == 0 || dom_[d].impl_last() < block_->size(dim, d));
    }
    map_.impl_apply(block_domain<dim>(*this));
  }

  Subset(Subset const& b)
    : block_ (&*b.block_),
      dom_ (b.dom_),
      map_ (b.map_)
  {
    map_.impl_apply(block_domain<dim>(*this));
  }

  ~Subset() VSIP_NOTHROW {}

  // Accessors.
  // The size of a Subset is the (total) size of its Domain(s), not
  // the size of the underlying block.
  length_type size() const VSIP_NOTHROW { return dom_.size();}
  length_type size(dimension_type block_d OVXX_UNUSED, dimension_type d) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(block_d == dim);
    OVXX_PRECONDITION(d < block_d);
    return dom_[d].size();
  }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return this->map_;}

  value_type get(index_type i) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(1, 0));
    return block_->get(dom_[0].impl_nth(i));
  }
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(2, 0));
    OVXX_PRECONDITION(j < this->size(2, 1));
    return block_->get(dom_[0].impl_nth(i),
		       dom_[1].impl_nth(j));
  }
  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < this->size(3, 0));
    OVXX_PRECONDITION(j < this->size(3, 1));
    OVXX_PRECONDITION(k < this->size(3, 2));
    return block_->get(dom_[0].impl_nth(i),
		       dom_[1].impl_nth(j),
		       dom_[2].impl_nth(k));
  }

  void put(index_type i, value_type val) VSIP_NOTHROW
  {
    block_->put(dom_[0].impl_nth(i), val);
  }
  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  {
    block_->put(dom_[0].impl_nth(i),
		dom_[1].impl_nth(j), val);
  }
  void put(index_type i, index_type j, index_type k, value_type val) VSIP_NOTHROW
  {
    block_->put(dom_[0].impl_nth(i),
		dom_[1].impl_nth(j),
		dom_[2].impl_nth(k), val);
  }


  B const &block() const { return *this->block_;}
  B &block() { return *this->block_;}
  Domain<dim> const &domain() const { return this->dom_;}

  // Lvalue interface
  reference_type ref(index_type i) VSIP_NOTHROW
  {
    return block_->ref(dom_[0].impl_nth(i));
  }
  reference_type ref(index_type i, index_type j) VSIP_NOTHROW
  {
    return block_->ref(dom_[0].impl_nth(i),
		       dom_[1].impl_nth(j));
  }
  reference_type ref(index_type i, index_type j, index_type k) VSIP_NOTHROW
  {
    return block_->ref(dom_[0].impl_nth(i),
		       dom_[1].impl_nth(j),
		       dom_[2].impl_nth(k));
  }

  parallel::par_ll_pbuf_type impl_ll_pbuf() VSIP_NOTHROW
  { return block_->impl_ll_pbuf();}

  stride_type offset() VSIP_NOTHROW
  {
    stride_type offset = block_->offset();
    for (dimension_type d=0; d<dim; ++d)
      offset += dom_[d].first() * block_->stride(dim, d);
    return offset;
  }

  ptr_type ptr() VSIP_NOTHROW
  { 
    ptr_type ptr = block_->ptr();
    for (dimension_type d=0; d<dim; ++d)
      ptr = storage::offset(ptr, dom_[d].first() * block_->stride(dim, d));
    return ptr;
  }

  const_ptr_type ptr() const VSIP_NOTHROW
  { 
    ptr_type ptr = block_->ptr();
    for (dimension_type d=0; d<dim; ++d)
      ptr = storage::offset(ptr, dom_[d].first() * block_->stride(dim, d));
    return ptr;
  }

  stride_type stride(dimension_type Dim OVXX_UNUSED, dimension_type d)
     const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    return block_->stride(dim, d) * dom_[d].stride();
  }

private:
  // Data members.
  typename block_traits<B>::ptr_type block_;
  Domain<dim> dom_;
  map_type    map_;
};

} // namespace ovxx::expr

template <typename B>
struct block_traits<expr::Subset<B> > : by_value_traits<expr::Subset<B> >
{};

template <typename B>
struct is_modifiable_block<expr::Subset<B> > : is_modifiable_block<B> {};

template <typename B>
struct distributed_local_block<expr::Subset<B> >
{
  typedef expr::Subset<typename distributed_local_block<B>::type> type;
  typedef expr::Subset<typename distributed_local_block<B>::proxy_type> proxy_type;
};

namespace detail
{

template <typename B>
expr::Subset<typename distributed_local_block<B>::type>
get_local_block(expr::Subset<B> const& block)
{
  typedef typename distributed_local_block<B>::type super_type;
  typedef expr::Subset<super_type> local_block_type;

  dimension_type const dim = expr::Subset<B>::dim;

  index_type sb = block.map().impl_rank_from_proc(local_processor());
  Domain<dim> dom;

  if (sb != no_subblock)
    dom = expr::detail::subset_parent_local_domain<expr::Subset<B> >
      ::parent_local_domain(block, sb);
  else
    dom = empty_domain<dim>();

  typename block_traits<super_type>::plain_type
    super_block = get_local_block(block.block());

  return local_block_type(dom, super_block);
}

template <typename B>
expr::Subset<typename distributed_local_block<B>::proxy_type>
get_local_proxy(expr::Subset<B> const &block, index_type sb)
{
  static dimension_type const dim = B::dim;
  typedef typename distributed_local_block<B>::proxy_type super_type;
  typedef expr::Subset<super_type> local_proxy_type;

  index_type super_sb = expr::detail::subset_map<dim, typename B::map_type>::
    parent_subblock(block.block().map(), block.domain(), sb);

  Domain<dim> l_dom = block.block().map().
    impl_local_from_global_domain(sb,
				  block.domain());

  typename block_traits<super_type>::plain_type
    super_block = get_local_proxy(block.block(), super_sb);

  return local_proxy_type(l_dom, super_block);
}

template <typename B>
void assert_local(expr::Subset<B> const &, index_type) {}

} // namespace ovxx::expr

template <typename B>
struct lvalue_factory_type<expr::Subset<B> >
{
  typedef typename lvalue_factory_type<B>::
    template rebind<expr::Subset<B> >::type type;
  template <typename O>
  struct rebind 
  {
    typedef typename lvalue_factory_type<B>::
      template rebind<O>::type type;
  };
};

} // namespace ovxx

namespace vsip
{

/// Set packing to unknown
template <typename B>
struct get_block_layout<ovxx::expr::Subset<B> >
{
  static dimension_type const dim = B::dim;

  typedef typename get_block_layout<B>::order_type   order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename B>
struct supports_dda<ovxx::expr::Subset<B> >
{ static bool const value = supports_dda<B>::value;};

} // namespace vsip

#endif
