//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_sliced_hpp_
#define ovxx_expr_sliced_hpp_

#include <ovxx/support.hpp>
#include <vsip/impl/local_map.hpp>
#include <ovxx/parallel/service.hpp>

namespace ovxx
{
namespace expr
{
namespace detail
{

template <typename M, dimension_type D>
struct sliced_map {};

template <typename M, dimension_type S1, dimension_type S2>
struct sliced2_map {};

template <dimension_type D, dimension_type S>
struct sliced_map<Replicated_map<D>, S>
{
  typedef Replicated_map<D - 1> type;
  static type convert_map(Replicated_map<D> const&, index_type) { return type();}
}; 

template <dimension_type D, dimension_type S>
struct sliced_map<parallel::local_or_global_map<D>, S>
{
  typedef parallel::local_or_global_map<D - 1> type;
  static type convert_map(parallel::local_or_global_map<D> const&, index_type)
  { return type();}
}; 

template <dimension_type S>
struct sliced_map<Local_map, S>
{
  typedef Local_map type;
  static type convert_map(Local_map const&, index_type) { return type();}
}; 

template <dimension_type D, dimension_type S>
struct sliced_map<parallel::scalar_map<D>, S>
{
  typedef parallel::scalar_map<D - 1> type;
  static type convert_map(parallel::scalar_map<D> const&, index_type)
  { return type();}
}; 

template <typename D0, typename D1, typename D2, dimension_type S>
struct sliced_map<Map<D0, D1, D2>, S>
{
  typedef typename parallel::map_project_1<S, Map<D0, D1, D2> >::type type;

  static type convert_map(Map<D0, D1, D2> const& map, index_type i)
  {
    return parallel::map_project_1<S, Map<D0, D1, D2> >::project(map, i);
  }

  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type i,
				    index_type sb)
  {
    return parallel::map_project_1<S, Map<D0, D1, D2> >::parent_subblock(map, i, sb);
  }
}; 

template <dimension_type D, dimension_type S1, dimension_type S2>
struct sliced2_map<Replicated_map<D>, S1, S2>
{
  typedef Replicated_map<D - 2> type;
  static type convert_map(Replicated_map<D> const&, index_type, index_type) { return type();}
}; 

template <dimension_type S1, dimension_type S2>
struct sliced2_map<Local_map, S1, S2>
{
  typedef Local_map type;
  static type convert_map(Local_map const&, index_type, index_type) { return type();}
}; 

template <typename D0, typename D1, typename D2,
	  dimension_type S1, dimension_type S2>
struct sliced2_map<Map<D0, D1, D2>, S1, S2>
{
  typedef parallel::map_project_2<S1, S2, Map<D0, D1, D2> > project_t;
  typedef typename project_t::type type;

  static type convert_map(Map<D0, D1, D2> const& map,
			  index_type idx0,
			  index_type idx1)
  { return project_t::project(map, idx0, idx1);}

  static index_type parent_subblock(Map<D0, D1, D2> const& map,
				    index_type idx0,
				    index_type idx1,
				    index_type sb)
  { return project_t::parent_subblock(map, idx0, idx1, sb);}
}; 

} // namespace ovxx::expr::detail

/// The Sliced class binds one of the indices of the underlying
/// N-dimensional block to a constant, producing an N-1-dimensional
/// block.  N must be 2 or 3, and the index chosen must be allowed for
/// the underlying block.
template <typename Block, dimension_type D> class Sliced;

template <typename Block, dimension_type D> 
class Sliced_base : ct_assert<(Block::dim >= 2)>,
                    ovxx::detail::nonassignable
{
public:
  static dimension_type const dim = Block::dim - 1;
  typedef typename Block::value_type value_type;
  typedef value_type&                reference_type;
  typedef value_type const&          const_reference_type;
  typedef typename detail::sliced_map<typename Block::map_type, D>::type map_type;

  Sliced_base(Sliced_base const& sb) VSIP_NOTHROW
    : map_(sb.map_), block_(&*sb.block_), index_(sb.index_)
  { map_.impl_apply(block_domain<dim>(*this)); }
  Sliced_base(Block &block, index_type i) VSIP_NOTHROW
    : map_(detail::sliced_map<typename Block::map_type, D>::convert_map(block.map(), i)),
      block_(&block),
      index_(i)
  { map_.impl_apply(block_domain<dim>(*this));}
  ~Sliced_base() VSIP_NOTHROW {}

  map_type const& map() const { return map_;}

  // Accessors.
  // The total size of a sliced block is the total size of the underlying
  // block, divided by the size of the bound index.
  length_type size() const VSIP_NOTHROW
  { return index_ == no_index ? 0 : block_->size() / block_->size(Block::dim, D);}
  length_type size(dimension_type block_d, dimension_type d) const VSIP_NOTHROW
  { return index_ == no_index ? 0 :
      block_->size(block_d + 1, ovxx::detail::compare<dimension_type, D>() > d ? d : d + 1);
  }
  // These are noops as Sliced is held by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  Block const &block() const { return *this->block_;}
  index_type index() const { return this->index_;}

  // Support Direct_data interface.
public:
  typedef storage_traits<value_type, get_block_layout<Block>::storage_format> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;

  ovxx::parallel::ll_pbuf_type impl_ll_pbuf() VSIP_NOTHROW
  { return block_->impl_ll_pbuf(); }

  stride_type offset() VSIP_NOTHROW
  {
    return block_->offset() + index_*block_->stride(Block::dim, D);
  }

  ptr_type ptr() VSIP_NOTHROW
  {
    return storage::offset(block_->ptr(),
			   index_*block_->stride(Block::dim, D));
  }
  const_ptr_type ptr() const VSIP_NOTHROW
  {
    return storage::offset(block_->ptr(),
			   index_*block_->stride(Block::dim, D));
  }
  stride_type stride(dimension_type Dim OVXX_UNUSED, dimension_type d)
     const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    return block_->stride(Block::dim,
			ovxx::detail::compare<dimension_type, D>() > d ? d : d+1);
  }

protected:
  map_type map_;
  typename block_traits<Block>::ptr_type block_;
  index_type const index_;
};

template <typename Block>
class Sliced<Block, 0> : public Sliced_base<Block, 0>
{
  typedef Sliced_base<Block, 0> Base;
public:
  typedef typename Base::value_type value_type;
  typedef typename Base::reference_type reference_type;

  Sliced(Sliced const& sb) VSIP_NOTHROW : Base(sb) {}
  Sliced(Block &block, index_type i) VSIP_NOTHROW : Base(block, i) {}

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW 
  { return this->block_->get(this->index_, i);}
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  { return this->block_->get(this->index_, i, j);}

  void put(index_type i, value_type val) VSIP_NOTHROW
  { this->block_->put(this->index_, i, val);}
  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  { this->block_->put(this->index_, i, j, val);}

  reference_type ref(index_type i) VSIP_NOTHROW
  { return this->block_->ref(this->index_, i);}
  reference_type ref(index_type i, index_type j) VSIP_NOTHROW
  { return this->block_->ref(this->index_, i, j);}
};

template <typename Block>
class Sliced<Block, 1> : public Sliced_base<Block, 1>
{
  typedef Sliced_base<Block, 1> Base;
public:
  typedef typename Base::value_type value_type;
  typedef typename Base::reference_type reference_type;

  Sliced(Sliced const& sb) VSIP_NOTHROW : Base(sb) {}
  Sliced(Block &block, index_type i) VSIP_NOTHROW : Base(block, i) {}

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW 
  { return this->block_->get(i, this->index_);}
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  { return this->block_->get(i, this->index_, j);}

  void put(index_type i, value_type val) VSIP_NOTHROW
  { this->block_->put(i, this->index_, val);}
  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  { this->block_->put(i, this->index_, j, val);}

  reference_type ref(index_type i) const VSIP_NOTHROW 
  { return this->block_->ref(i, this->index_);}
  reference_type ref(index_type i, index_type j) const VSIP_NOTHROW
  { return this->block_->ref(i, this->index_, j);}
};

template <typename Block>
class Sliced<Block, 2> : public Sliced_base<Block, 2>
{
  typedef Sliced_base<Block, 2> Base;
public:
  typedef typename Base::value_type value_type;
  typedef typename Base::reference_type reference_type;

  Sliced(Sliced const& sb) VSIP_NOTHROW : Base(sb) {}
  Sliced(Block &block, index_type i) VSIP_NOTHROW : Base(block, i) {}

  // Data accessors.
  value_type get(index_type i, index_type j) const VSIP_NOTHROW
  { return this->block_->get(i, j, this->index_);}

  void put(index_type i, index_type j, value_type val) VSIP_NOTHROW
  { this->block_->put(i, j, this->index_, val);}

  reference_type ref(index_type i, index_type j) const VSIP_NOTHROW
  { return this->block_->ref(i, j, this->index_);}
};

} // namespace ovxx::expr

template <typename B, dimension_type D>
struct block_traits<expr::Sliced<B, D> > : by_value_traits<expr::Sliced<B, D> >
{};

template <typename B, dimension_type D>
struct is_modifiable_block<expr::Sliced<B, D> > : is_modifiable_block<B>
{};

namespace expr
{

/// The Sliced2_block class binds two of the indices of the underlying
/// N-dimensional block to a constant, producing an N-2-dimensional
/// block.  N must be >= 3, and the index chosen must be allowed for
/// the underlying block.
template <typename Block, dimension_type D1, dimension_type D2> 
class Sliced2;

template <typename Block, dimension_type D1, dimension_type D2>
class Sliced2_base : ct_assert<(Block::dim > D2 && D2 > D1)>,
                     ovxx::detail::nonassignable
{
public:
  static dimension_type const dim = Block::dim - 2;
  typedef typename Block::value_type value_type;
  typedef value_type&       reference_type;
  typedef value_type const& const_reference_type;
  typedef typename detail::sliced2_map<typename Block::map_type, D1, D2>::type map_type;

  Sliced2_base(Sliced2_base const& sb) VSIP_NOTHROW
    : map_(sb.map_), block_(&*sb.block_), index1_(sb.index1_), index2_(sb.index2_)
  { map_.impl_apply(block_domain<dim>(*this)); }
  Sliced2_base(Block &block, index_type i, index_type j) VSIP_NOTHROW
    : map_(detail::sliced2_map<typename Block::map_type,
	   D1,
	   D2>::convert_map(block.map(), i, j)),
      block_(&block), index1_(i), index2_(j)
  { map_.impl_apply(block_domain<dim>(*this)); }
  ~Sliced2_base() VSIP_NOTHROW {}

  map_type const& map() const { return map_;}

  // Accessors.
  // The total size of a sliced block is the total size of the underlying
  // block, divided by the size of the bound index.
  length_type size() const VSIP_NOTHROW
  { return block_->size() / block_->size(Block::dim, D1) / block_->size(Block::dim, D2);}
  length_type size(dimension_type block_d, dimension_type d) const VSIP_NOTHROW
  { return block_->size(block_d + 2,
		      ovxx::detail::compare<dimension_type, D1>() > d     ? d   :
		      ovxx::detail::compare<dimension_type, D2>() > (d+1) ? d+1 : d+2);
  }
  // These are noops as Sliced2 is helt by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  Block const &block() const { return *this->block_;}
  index_type index1() const { return this->index1_;}
  index_type index2() const { return this->index2_;}

  // Support Direct_data interface.
public:
  typedef storage_traits<value_type, get_block_layout<Block>::storage_format> storage;
  typedef typename storage::ptr_type ptr_type;
  typedef typename storage::const_ptr_type const_ptr_type;

  ovxx::parallel::ll_pbuf_type impl_ll_pbuf() VSIP_NOTHROW
  { return block_->impl_ll_pbuf(); }

  stride_type offset() VSIP_NOTHROW
  {
    return block_->offset()
	 + index1_*block_->stride(Block::dim, D1)
	 + index2_*block_->stride(Block::dim, D2);
  }

  ptr_type ptr() VSIP_NOTHROW
  {
    return storage::offset(block_->ptr(),
			   + index1_*block_->stride(Block::dim, D1)
			   + index2_*block_->stride(Block::dim, D2));
  }

  const_ptr_type ptr() const VSIP_NOTHROW
  {
    return storage::offset(block_->ptr(),
			   + index1_*block_->stride(Block::dim, D1)
			   + index2_*block_->stride(Block::dim, D2));
  }

  stride_type stride(dimension_type Dim OVXX_UNUSED, dimension_type d)
     const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(Dim == dim && d<dim);
    return block_->stride(Block::dim,
			ovxx::detail::compare<dimension_type, D1>() > d     ? d   :
			ovxx::detail::compare<dimension_type, D2>() > (d+1) ? d+1 : d+2);
  }

protected:
  map_type map_;
  typename block_traits<Block>::ptr_type block_;
  index_type const index1_;
  index_type const index2_;
};

template <typename Block>
class Sliced2<Block, 0, 1> : public Sliced2_base<Block, 0, 1>
{
  typedef Sliced2_base<Block, 0, 1> Base;
public:
  typedef typename Base::value_type value_type;
  typedef typename Base::reference_type reference_type;

  Sliced2(Sliced2 const& sb) VSIP_NOTHROW : Base(sb) {}
  Sliced2(Block &block, index_type i, index_type j) VSIP_NOTHROW 
    : Base(block, i, j)
  {}

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW 
  { return this->block_->get(this->index1_, this->index2_, i);}

  void put(index_type i, value_type val) VSIP_NOTHROW
  { this->block_->put(this->index1_, this->index2_, i, val);}

  reference_type ref(index_type i) const VSIP_NOTHROW 
  { return this->block_->ref(this->index1_, this->index2_, i);}
};

template <typename Block>
class Sliced2<Block, 0, 2> : public Sliced2_base<Block, 0, 2>
{
  typedef Sliced2_base<Block, 0, 2> Base;
public:
  typedef typename Base::value_type value_type;
  typedef typename Base::reference_type reference_type;

  Sliced2(Sliced2 const& sb) VSIP_NOTHROW : Base(sb) {}
  Sliced2(Block &block, index_type i, index_type j) VSIP_NOTHROW 
    : Base(block, i, j)
  {}

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW 
  { return this->block_->get(this->index1_, i, this->index2_);}

  void put(index_type i, value_type val) VSIP_NOTHROW
  { this->block_->put(this->index1_, i, this->index2_, val);}

  reference_type ref(index_type i) const VSIP_NOTHROW 
  { return this->block_->ref(this->index1_, i, this->index2_);}
};

template <typename Block>
class Sliced2<Block, 1, 2> : public Sliced2_base<Block, 1, 2>
{
  typedef Sliced2_base<Block, 1, 2> Base;
public:
  typedef typename Base::value_type value_type;
  typedef typename Base::reference_type reference_type;

  Sliced2(Sliced2 const& sb) VSIP_NOTHROW : Base(sb) {}
  Sliced2(Block &block, index_type i, index_type j) VSIP_NOTHROW 
    : Base(block, i, j)
  {}

  // Data accessors.
  value_type get(index_type i) const VSIP_NOTHROW 
  { return this->block_->get(i, this->index1_, this->index2_);}

  void put(index_type i, value_type val) VSIP_NOTHROW
  { this->block_->put(i, this->index1_, this->index2_, val);}

  reference_type ref(index_type i) const VSIP_NOTHROW 
  { return this->block_->ref(i, this->index1_, this->index2_);}
};



template <typename       Tuple,
	  dimension_type NumDim,
	  dimension_type FixedDim>
struct Sliced_order;

template <dimension_type Dim0,
	  dimension_type Dim1,
	  dimension_type Dim2,
	  dimension_type FixedDim>
struct Sliced_order<tuple<Dim0, Dim1, Dim2>, 2, FixedDim>
{
  typedef row1_type type;
  static bool const unit_stride_preserved = (FixedDim != Dim1);
};

template <dimension_type Dim0,
	  dimension_type Dim1,
	  dimension_type Dim2,
	  dimension_type FixedDim>
struct Sliced_order<tuple<Dim0, Dim1, Dim2>, 3, FixedDim>
{
  typedef typename
  conditional<FixedDim == Dim0,
    typename conditional<(Dim2 > Dim1), row2_type, col2_type>::type,
    typename conditional<FixedDim == Dim1,
      typename conditional<(Dim2 > Dim0), row2_type, col2_type>::type,
      typename conditional<(Dim1 > Dim0), row2_type, col2_type>::type>::type>::type
    type;
  static bool const unit_stride_preserved = (FixedDim != Dim2);
};

} // namespace ovxx::expr

template <typename B, dimension_type D1, dimension_type D2>
struct block_traits<expr::Sliced2<B, D1, D2> >
  : by_value_traits<expr::Sliced2<B, D1, D2> >
{};

template <typename B, dimension_type D1, dimension_type D2>
struct is_modifiable_block<expr::Sliced2<B, D1, D2> > : is_modifiable_block<B>
{};

template <typename B, dimension_type D>
struct distributed_local_block<expr::Sliced<B, D> >
{
  typedef expr::Sliced<typename distributed_local_block<B>::type, D> type;
  typedef expr::Sliced<typename distributed_local_block<B>::proxy_type, D> proxy_type;
};

template <typename B, dimension_type D1, dimension_type D2> 
struct distributed_local_block<expr::Sliced2<B, D1, D2> >
{
  typedef expr::Sliced2<typename distributed_local_block<B>::type, D1, D2> type;
  typedef expr::Sliced2<typename distributed_local_block<B>::proxy_type, D1, D2> 
    proxy_type;
};

namespace detail
{

template <typename B, dimension_type D>
expr::Sliced<typename distributed_local_block<B>::type, D>
get_local_block(expr::Sliced<B, D> const& block)
{
  typedef expr::Sliced<typename distributed_local_block<B>::type, D> local_block_type;

  // This conversion is only valid if the local processor holds
  // the subblock containing the slice.

  index_type idx;
  if (block.map().subblock() != no_subblock)
    idx = block.block().map().
      impl_local_from_global_index(D, block.index());
  else
    idx = no_index;

  return local_block_type(get_local_block(block.block()), idx);
}

template <typename B, dimension_type D>
expr::Sliced<typename distributed_local_block<B>::proxy_type, D>
get_local_proxy(expr::Sliced<B, D> const &block, index_type sb)
{
  typedef typename distributed_local_block<B>::proxy_type super_type;
  typedef expr::Sliced<super_type, D> local_proxy_type;

  index_type super_sb = expr::detail::sliced_map<typename B::map_type, D>::
    parent_subblock(block.block().map(), block.index(), sb);

  index_type l_idx = block.block().map().
      impl_local_from_global_index(D, block.index());

  typename block_traits<super_type>::plain_type
    super_block = get_local_proxy(block.block(), super_sb);
  return local_proxy_type(super_block, l_idx);
}

template <typename B, dimension_type D1, dimension_type D2>
expr::Sliced2<typename distributed_local_block<B>::type, D1, D2>
get_local_block(expr::Sliced2<B, D1, D2> const &block)
{
  typedef expr::Sliced2<typename distributed_local_block<B>::type, D1, D2>
    local_block_type;

  index_type idx1 = block.block().map().
    impl_local_from_global_index(D1, block.index1());
  index_type idx2 = block.block().map().
    impl_local_from_global_index(D2, block.index2());

  return local_block_type(get_local_block(block.block()), idx1, idx2);
}

template <typename B, dimension_type D1, dimension_type D2>
expr::Sliced2<typename distributed_local_block<B>::proxy_type, D1, D2>
get_local_proxy(expr::Sliced2<B, D1, D2> const& block, index_type sb)
{
  typedef typename distributed_local_block<B>::proxy_type super_type;
  typedef expr::Sliced2<super_type, D1, D2> local_block_type;

  index_type l_idx1 = block.block().map().
    impl_local_from_global_index(D1, block.index1());
  index_type l_idx2 = block.block().map().
    impl_local_from_global_index(D2, block.index2());

  index_type super_sb = expr::detail::sliced2_map<typename B::map_type, D1, D2>::
    parent_subblock(block.block().map(), block.index1(), block.index2(), sb);

  typename block_traits<super_type>::plain_type
    super_block = get_local_proxy(block.block(), super_sb);

  return local_block_type(get_local_block(block.block()), l_idx1, l_idx2);
}

template <typename B, dimension_type D>
void assert_local(expr::Sliced<B, D> const &, index_type) {}

template <typename B, dimension_type D1, dimension_type D2>
void assert_local(expr::Sliced2<B, D1, D2> const &, index_type) {}

} // namespace ovxx::detail

template <typename B, dimension_type D>
struct lvalue_factory_type<expr::Sliced<B, D>, D>
{
  typedef typename lvalue_factory_type<B, D>
    ::template rebind<expr::Sliced<B, D> >::type type;
  template <typename O>
  struct rebind 
  {
    typedef typename lvalue_factory_type<B, D>::
      template rebind<O>::type type;
  };
};

template <typename B, dimension_type D1, dimension_type D2>
struct lvalue_factory_type<expr::Sliced2<B, D1, D2>, 1>
{
  typedef typename lvalue_factory_type<B, 3>
  ::template rebind<expr::Sliced2<B, D1, D2> >::type type;
  template <typename O>
  struct rebind 
  {
    typedef typename lvalue_factory_type<B, 3>::
      template rebind<O>::type type;
  };
};

} // namespace ovxx

namespace vsip
{
/// Dimension is reduced by 1, dimension-order is preserved
/// (with the fixed dimension being taken out).
/// unit-stride is preserved if the fixed dimension is not the 
/// minor dimension, set to unknown otherwise.
template <typename B, dimension_type D>
struct get_block_layout<ovxx::expr::Sliced<B, D> >
{
private:
  typedef ovxx::expr::Sliced_order<typename get_block_layout<B>::order_type,
				   B::dim,
				   D> sbo_type;

public:

  static dimension_type const dim = B::dim-1;

  typedef typename sbo_type::type order_type;
  static pack_type const packing =
    is_packing_unit_stride<get_block_layout<B>::packing>::value &&
    sbo_type::unit_stride_preserved ?
    unit_stride : any_packing;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename B, dimension_type D>
struct supports_dda<ovxx::expr::Sliced<B, D> >
{ static bool const value = supports_dda<B>::value;};

template <typename B, dimension_type D1, dimension_type D2> 
struct get_block_layout<ovxx::expr::Sliced2<B, D1, D2> >
{
private:
  typedef typename get_block_layout<B>::order_type par_order_type;

public:
  static dimension_type const dim = B::dim-2;

  typedef row1_type order_type;

  static pack_type const packing = 
    is_packing_unit_stride<get_block_layout<B>::packing>::value &&
    D1 != par_order_type::impl_dim2 && D2 != par_order_type::impl_dim2 ?
    unit_stride : any_packing;
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename B, dimension_type D1, dimension_type D2>
struct supports_dda<ovxx::expr::Sliced2<B, D1, D2> >
{ static bool const value = supports_dda<B>::value;};

} // namespace vsip

#endif
