/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/us_block.hpp
    @author  Jules Bergmann
    @date    2006-01-31
    @brief   VSIPL++ Library: User-storage block class.

*/

#ifndef VSIP_CORE_US_BLOCK_HPP
#define VSIP_CORE_US_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/block_traits.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{ 

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT = Layout<Dim, 
					  typename impl::Row_major<Dim>::type,
					  Stride_unit_dense,
					  Cmplx_inter_fmt>,
	  typename       MapT = Local_map>
class Us_block
  : impl::Non_copyable,
    public impl::Ref_count<Us_block<Dim, T, LayoutT, MapT> >
{
  // Compile-time values and types.
public:
  static dimension_type const dim = Dim;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef typename LayoutT::order_type order_type;
  typedef MapT                         map_type;

  // Enable Direct_data access to data.
  template <typename, typename, typename>
  friend class impl::data_access::Low_level_data_access;

  // Implementation types.
public:
  typedef LayoutT                                    layout_type;
  typedef impl::Applied_layout<layout_type>          applied_layout_type;
  typedef Storage<typename LayoutT::complex_type, T> storage_type;

  // Constructors and destructor.
public:
  Us_block(Domain<Dim> const&          dom,
	   typename storage_type::type ptr,
	   MapT const&                 map = MapT())
  VSIP_THROW((std::bad_alloc))
    : layout_ (dom),
      storage_(ptr),
      map_    (map)
  { assert(!storage_type::is_null(storage_)); }

  ~Us_block() VSIP_NOTHROW {}

  // Data accessors.
public:

  // 1-dimensional accessors
  T get(index_type idx) const VSIP_NOTHROW
  {
    assert(idx < size());
    return storage_type::get(storage_, idx);
  }

  void put(index_type idx, T val) VSIP_NOTHROW
  {
    assert(idx < size());
    storage_type::put(storage_, idx, val);
  }

  // 2-diminsional get/put
  T    get(index_type idx0, index_type idx1) const VSIP_NOTHROW
    { return storage_type::get(storage_, layout_.index(idx0, idx1)); }
  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
    { storage_type::put(storage_, layout_.index(idx0, idx1), val); }

  // 3-diminsional get/put
  T    get(index_type idx0, index_type idx1, index_type idx2)
    const VSIP_NOTHROW
    { return storage_type::get(storage_, layout_.index(idx0, idx1, idx2)); }
  void put(index_type idx0, index_type idx1, index_type idx2, T val)
    VSIP_NOTHROW
    { storage_type::put(storage_, layout_.index(idx0, idx1, idx2), val); }

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW;
  length_type size(dimension_type D, dimension_type d) const VSIP_NOTHROW;
  MapT const& map() const VSIP_NOTHROW { return map_;}

  // Support Direct_data interface.
public:
  typedef typename storage_type::type       data_type;
  typedef typename storage_type::const_type const_data_type;

  data_type       impl_data()       VSIP_NOTHROW { return storage_; }
  const_data_type impl_data() const VSIP_NOTHROW { return storage_; }
  stride_type impl_stride(dimension_type D, dimension_type d)
    const VSIP_NOTHROW;

  // Member Data.
public:
  applied_layout_type layout_;
  data_type           storage_;
  map_type            map_;
};



/// Specialize block layout trait for Us_block blocks.

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT,
	  typename       Map>
struct Block_layout<Us_block<Dim, T, LayoutT, Map> >
{
  static dimension_type const dim = Dim;

  typedef Direct_access_tag access_type;
  typedef typename LayoutT::order_type   order_type;
  typedef typename LayoutT::pack_type    pack_type;
  typedef typename LayoutT::complex_type complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> layout_type;
};

/// Specialize lvalue accessor trait for Us_blocks.  Us_block provides
/// direct lvalue accessors via impl_ref.

template <typename BlockT,
	  bool     use_proxy =
	              Is_complex<typename BlockT::value_type>::value &&
                      Type_equal<typename Block_layout<BlockT>::complex_type,
				 Cmplx_split_fmt>::value>
struct Us_block_lvalue_factory_type;

template <typename BlockT>
struct Us_block_lvalue_factory_type<BlockT, false>
{
  typedef True_lvalue_factory<BlockT> type;
  template <typename OtherBlock>
  struct Rebind {
    typedef True_lvalue_factory<OtherBlock> type;
  };
};

template <typename BlockT>
struct Us_block_lvalue_factory_type<BlockT, true>
{
  typedef Proxy_lvalue_factory<BlockT> type;
  template <typename OtherBlock>
  struct Rebind {
    typedef Proxy_lvalue_factory<OtherBlock> type;
  };
};

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT>
struct Lvalue_factory_type<Us_block<Dim, T, LayoutT, Local_map> >
  : public Us_block_lvalue_factory_type<Us_block<Dim, T, LayoutT, Local_map> >
{};



/// Specialize Distributed_local_block traits class for Us_block.

/// For a serial map, distributed block and local block are the same.

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT>
struct Distributed_local_block<Us_block<Dim, T, LayoutT, Local_map> >
{
  typedef Us_block<Dim, T, LayoutT, Local_map> type;
};



/// For a distributed map, local block has a serial map.

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT,
	  typename       MapT>
struct Distributed_local_block<Us_block<Dim, T, LayoutT, MapT> >
{
  // We could determine the local block by just chaning the map
  // to serial:
  //   typedef Us_block<Dim, T, LayoutT, Local_map> type;

  // However, to be safe, we'll extract it from the block itself:
  // (local_block is set in the base class Distributed_block.)
  typedef typename Us_block<Dim, T, LayoutT, MapT>::local_block_type type;
};



/// Overload of get_local_block for Us_block with serial map.

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
Us_block<Dim, T, OrderT, Local_map>&
get_local_block(
  Us_block<Dim, T, OrderT, Local_map>& block)
{
  return block;
}

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
Us_block<Dim, T, OrderT, Local_map> const&
get_local_block(
  Us_block<Dim, T, OrderT, Local_map> const& block)
{
  return block;
}



/// Overload of get_local_block for Us_block with distributed map.

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
Us_block<Dim, T, OrderT, Local_map>&
get_local_block(
  Us_block<Dim, T, OrderT, MapT> const& block)
{
  return block.get_local_block();
}



/// Assert that subblock is local to block (overload).

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
void
assert_local(
  Us_block<Dim, T, OrderT, Local_map> const& block,
  index_type                              sb)
{
  assert(sb == 0);
}



/// Assert that subblock is local to block (overload).

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
void
assert_local(
  Us_block<Dim, T, OrderT, MapT> const& block,
  index_type                         sb)
{
  block.assert_local(sb);
}



/// Specialize Is_simple_distributed_block traits class for Us_block.

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
struct Is_simple_distributed_block<Us_block<Dim, T, OrderT, MapT> >
{
  static bool const value = true;
};



#if VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES==0

/// Specialize Combine_return_type for Us_block block leaves.

template <typename       CombineT,
	  dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
struct Combine_return_type<CombineT, Us_block<Dim, T, OrderT, MapT> >
{
  typedef Us_block<Dim, T, OrderT, MapT> block_type;
  typedef typename CombineT::template return_type<block_type>::type
		type;
  typedef typename CombineT::template tree_type<block_type>::type
		tree_type;
};



/// Specialize apply_combine for Us_block block leaves.

template <typename       CombineT,
	  dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
typename Combine_return_type<CombineT, Us_block<Dim, T, OrderT, MapT> >::type
apply_combine(
  CombineT const&                    combine,
  Us_block<Dim, T, OrderT, MapT> const& block)
{
  return combine.apply(block);
}
#endif



/***********************************************************************
  Definitions - Us_blocks
***********************************************************************/

/// Return the total size of the block.

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT,
	  typename       MapT>
length_type
Us_block<Dim, T, LayoutT, MapT>::size() const VSIP_NOTHROW
{
  length_type retval = layout_.size(0);
  for (dimension_type d=1; d<Dim; ++d)
    retval *= layout_.size(d);
  return retval;
}



/// Return the size of the block in a specific dimension.

/// Requires:
///   BLOCK_DIM selects which block-dimensionality (BLOCK_DIM == 1).
///   DIM is the dimension whose length to return (0 <= DIM < BLOCK_DIM).
/// Returns:
///   The size of dimension DIM.

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT,
	  typename       MapT>
inline
length_type
Us_block<Dim, T, LayoutT, MapT>::size(
  dimension_type block_dim,
  dimension_type d)
  const VSIP_NOTHROW
{
  assert((block_dim == 1 || block_dim == Dim) && (d < block_dim));
  return (block_dim == 1) ? this->size() : this->layout_.size(d);
}



// Requires:
//   DIM is a valid dimensionality supported by block (DIM == 1 or 2)
//   D is a dimension, less than DIM.
// Returns
//   The stride in dimension D, for dimensionality DIM.

template <dimension_type Dim,
	  typename       T,
	  typename       LayoutT,
	  typename       MapT>
inline
stride_type
Us_block<Dim, T, LayoutT, MapT>::impl_stride(
  dimension_type block_dim, dimension_type d)
  const VSIP_NOTHROW
{
  assert((block_dim == 1 || block_dim == Dim) && (d < block_dim));

  return (block_dim == 1) ? 1 : layout_.stride(d);
}


} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_US_BLOCK_HPP
