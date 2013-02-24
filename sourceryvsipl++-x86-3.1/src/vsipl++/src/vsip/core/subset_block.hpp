/* Copyright (c) 2005, 2006, 2008, 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   Define Subset_block.

#ifndef vsip_core_subset_block_hpp_
#define vsip_core_subset_block_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/map_fwd.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/storage.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <vsip/core/parallel/subset_map_decl.hpp>
#include <complex>

namespace vsip
{

namespace impl
{

/// Functor to create map for a subset block from its parent block's map.
/// (not a map class)
template <dimension_type Dim,
	  typename       MapT>
struct Subset_block_map
{
  typedef typename Map_subdomain<Dim, MapT>::type type;

  static type convert_map(MapT const&        map,
			  Domain<Dim> const& dom)
  {
    return Map_subdomain<Dim, MapT>::project(map, dom);
  }

  static index_type parent_subblock(MapT const&        map,
				    Domain<Dim> const& dom,
				    index_type         sb)
  {
    return Map_subdomain<Dim, MapT>::parent_subblock(map, dom, sb);
  }
};

/// The Subset_block class maps all accesses through a Domain instance
/// before forwarding to the underlying block.  Thus, for instance, if
/// a 1-dimensional vector of length N is the underlying block, and the
/// Domain instance in use is Domain<1>(N/2)*2, then the Subset_block class
/// will expose every other element of the underlying block.
template <typename Block>
class Subset_block : public Non_assignable
{
public:
  // Compile-time values and types.
  static dimension_type const dim = Block::dim;
  typedef typename Block::value_type value_type;
  typedef value_type&                reference_type;
  typedef value_type const&          const_reference_type;

  typedef Subset_block_map<dim, typename Block::map_type> map_functor;
  typedef typename map_functor::type map_type;

  // Constructors and destructors.
  Subset_block(Domain<dim> const& dom, Block &blk) VSIP_NOTHROW
    : blk_ (&blk),
      dom_ (dom),
      map_ (map_functor::convert_map(blk.map(), dom_))
  {
    // Sanity check that all of the Domain indices are within the
    // underlying block's range.  (If domain is empty, value
    // returned by impl_last() is not valid.)
    for (dimension_type d = 0; d < dim; d++)
    {
      assert(dom_[d].size() == 0 || dom_[d].first() < blk_->size(dim, d));
      assert(dom_[d].size() == 0 || dom_[d].impl_last() < blk_->size(dim, d));
    }
    map_.impl_apply(block_domain<dim>(*this));
  }

  Subset_block(Subset_block const& b)
    : blk_ (&*b.blk_),
      dom_ (b.dom_),
      map_ (b.map_)
  {
    map_.impl_apply(block_domain<dim>(*this));
  }

  ~Subset_block() VSIP_NOTHROW {}

  // Accessors.
  // The size of a Subset is the (total) size of its Domain(s), not
  // the size of the underlying block.
  length_type size() const VSIP_NOTHROW { return dom_.size();}
  length_type size(dimension_type block_d ATTRIBUTE_UNUSED, dimension_type d) const VSIP_NOTHROW
  {
    assert(block_d == dim);
    assert(d < block_d);
    return dom_[d].size();
  }

  // These are noops as Subset_block is held by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return this->map_;}

  value_type get(index_type i) const VSIP_NOTHROW
  {
    assert(i < this->size(1, 0));
    return blk_->get(dom_[0].impl_nth(i));
  }
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  {
    assert(i < this->size(2, 0));
    assert(j < this->size(2, 1));
    return blk_->get(dom_[0].impl_nth(i),
		     dom_[1].impl_nth(j));
  }
  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
  {
    assert(i < this->size(3, 0));
    assert(j < this->size(3, 1));
    assert(k < this->size(3, 2));
    return blk_->get(dom_[0].impl_nth(i),
		     dom_[1].impl_nth(j),
		     dom_[2].impl_nth(k));
  }

  void put(index_type i, value_type val) VSIP_NOTHROW
  {
    blk_->put(dom_[0].impl_nth(i), val);
  }
  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  {
    blk_->put(dom_[0].impl_nth(i),
	      dom_[1].impl_nth(j), val);
  }
  void put(index_type i, index_type j, index_type k, value_type val) VSIP_NOTHROW
  {
    blk_->put(dom_[0].impl_nth(i),
	      dom_[1].impl_nth(j),
	      dom_[2].impl_nth(k), val);
  }


  Block const& impl_block() const { return *this->blk_; }
  Block & impl_block() { return *this->blk_;}
  Domain<dim> const& impl_domain() const { return this->dom_; }


  // Lvalue interface
  reference_type impl_ref(index_type i) VSIP_NOTHROW
  {
    return blk_->impl_ref(dom_[0].impl_nth(i));
  }
  reference_type impl_ref(index_type i, index_type j) VSIP_NOTHROW
  {
    return blk_->impl_ref(dom_[0].impl_nth(i),
                          dom_[1].impl_nth(j));
  }
  reference_type impl_ref(index_type i, index_type j, index_type k) VSIP_NOTHROW
  {
    return blk_->impl_ref(dom_[0].impl_nth(i),
                          dom_[1].impl_nth(j),
                          dom_[2].impl_nth(k));
  }

  // Support Direct_data interface.
public:
  typedef impl::Storage<get_block_layout<Block>::storage_format, value_type>
		storage_type;
  typedef typename storage_type::type       ptr_type;
  typedef typename storage_type::const_type const_ptr_type;

  par_ll_pbuf_type impl_ll_pbuf() VSIP_NOTHROW
  { return blk_->impl_ll_pbuf(); }

  stride_type impl_offset() VSIP_NOTHROW
  {
    stride_type offset = blk_->impl_offset();
    for (dimension_type d=0; d<dim; ++d)
      offset += dom_[d].first() * blk_->stride(dim, d);
    return offset;
  }

  ptr_type ptr() VSIP_NOTHROW
  { 
    ptr_type ptr = blk_->ptr();
    for (dimension_type d=0; d<dim; ++d)
      ptr = storage_type::offset(ptr,
				 dom_[d].first() * blk_->stride(dim, d));
    return ptr;
  }

  const_ptr_type ptr() const VSIP_NOTHROW
  { 
    ptr_type ptr = blk_->ptr();
    for (dimension_type d=0; d<dim; ++d)
      ptr = storage_type::offset(ptr,
				 dom_[d].first() * blk_->stride(dim, d));
    return ptr;
  }

  stride_type stride(dimension_type Dim ATTRIBUTE_UNUSED, dimension_type d)
     const VSIP_NOTHROW
  {
    assert(Dim == dim && d<dim);
    return blk_->stride(dim, d) * dom_[d].stride();
  }

private:
  // Data members.
  typename View_block_storage<Block>::type blk_;
  Domain<dim> dom_;
  map_type    map_;
};

// Store Subset_blocks by-value.
template <typename Block>
struct View_block_storage<Subset_block<Block> >
  : By_value_block_storage<Subset_block<Block> >
{};


template <typename BlockT>
struct Block_root<Subset_block<BlockT> >
{
  typedef typename Block_root<BlockT>::type type;
};

template <typename       BlockT>
typename Block_root<Subset_block<BlockT> >::type const&
block_root(Subset_block<BlockT> const& block)
{
  return block_root(block.impl_block());
}

// Subset_block has impl_ref if the underlying block has impl_ref.
template <typename Block>
struct Lvalue_factory_type<Subset_block<Block> >
{
  typedef typename Lvalue_factory_type<Block>
    ::template Rebind<Subset_block<Block> >::type type;
  template <typename OtherBlock>
  struct Rebind {
    typedef typename Lvalue_factory_type<Block>
      ::template Rebind<OtherBlock>::type type;
  };
};

template <typename BlockT>
struct is_modifiable_block<Subset_block<BlockT> >
  : is_modifiable_block<BlockT>
{};

template <typename Block>
struct Distributed_local_block<Subset_block<Block> >
{
  typedef Subset_block<typename Distributed_local_block<Block>::type> type;
  typedef Subset_block<typename Distributed_local_block<Block>::proxy_type> proxy_type;
};

/// Helper class to translate a distributed subset into a local subset.
/// In general case, ask the parent block what the local subset should be.
template <typename BlockT,
	  typename MapT = typename BlockT::map_type>
struct Subset_parent_local_domain
{
  static dimension_type const dim = BlockT::dim;

  static Domain<dim> parent_local_domain(BlockT const& block, index_type sb)
  {
    return block.impl_block().map().impl_local_from_global_domain(sb,
					block.impl_domain());
  }
};



/// Specialize for Subset_map, where the local subset is currently larger than
/// what the parent thinks it should be.
template <typename       BlockT,
	  dimension_type Dim>
struct Subset_parent_local_domain<BlockT, Subset_map<Dim> >
{
  static dimension_type const dim = BlockT::dim;

  static Domain<dim> parent_local_domain(BlockT const& block, index_type sb)
  {
#if 0
    // consider asking Subset_map about parent's local block,
    // rather than asking the parent to recompute it.
    return block.map().template impl_parent_local_domain<dim>(sb);
#else
    index_type parent_sb = block.map().impl_parent_subblock(sb);
    return block.impl_block().map().impl_local_from_global_domain(parent_sb,
					block.impl_domain());
#endif
  }
};



template <typename Block>
Subset_block<typename Distributed_local_block<Block>::type>
get_local_block(
  Subset_block<Block> const& block)
{
  typedef typename Distributed_local_block<Block>::type super_type;
  typedef Subset_block<super_type>                      local_block_type;

  dimension_type const dim = Subset_block<Block>::dim;

  index_type sb = block.map().impl_rank_from_proc(local_processor());
  Domain<dim> dom;

  if (sb != no_subblock)
    dom = Subset_parent_local_domain<Subset_block<Block> >::
      parent_local_domain(block, sb);
  else
    dom = empty_domain<dim>();

  typename View_block_storage<super_type>::plain_type
    super_block = get_local_block(block.impl_block());

  return local_block_type(dom, super_block);
}



template <typename Block>
Subset_block<typename Distributed_local_block<Block>::proxy_type>
get_local_proxy(
  Subset_block<Block> const& block,
  index_type                    sb)
{
  static dimension_type const dim = Block::dim;

  typedef typename Distributed_local_block<Block>::proxy_type super_type;
  typedef Subset_block<super_type>                            local_proxy_type;

  index_type super_sb = Subset_block_map<dim, typename Block::map_type>::
    parent_subblock(block.impl_block().map(), block.impl_domain(), sb);

  Domain<dim> l_dom = block.impl_block().map().
    impl_local_from_global_domain(sb,
				  block.impl_domain());

  typename View_block_storage<super_type>::plain_type
    super_block = get_local_proxy(block.impl_block(), super_sb);

  return local_proxy_type(l_dom, super_block);
}

template <typename Block>
void
assert_local(
  Subset_block<Block> const& /*block*/,
  index_type                 /*sb*/)
{
}

#if VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES==0

/// Specialize Combine_return_type for Subset_block leaves.
template <typename       CombineT,
	  typename       Block>
struct Combine_return_type<CombineT, Subset_block<Block> >
{
  typedef Subset_block<Block> block_type;
  typedef typename CombineT::template return_type<block_type>::type
		type;
  typedef typename CombineT::template tree_type<block_type>::type
		tree_type;
};



/// Specialize apply_combine for Subset_block leaves.
template <typename       CombineT,
	  typename       Block>
typename Combine_return_type<CombineT, Subset_block<Block> >::type
apply_combine(
  CombineT const&               combine,
  Subset_block<Block> const& block)
{
  return combine.apply(block);
}

#endif

} // namespace vsip::impl

/// Set packing to unknown
template <typename Block>
struct get_block_layout<impl::Subset_block<Block> >
{
  static dimension_type const dim = Block::dim;

  typedef typename get_block_layout<Block>::order_type   order_type;
  static pack_type const packing = any_packing;
  static storage_format_type const storage_format = get_block_layout<Block>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename Block>
struct supports_dda<impl::Subset_block<Block> >
{ static bool const value = supports_dda<Block>::value;};

} // namespace vsip

#endif
