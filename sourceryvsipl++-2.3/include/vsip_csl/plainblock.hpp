/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/plainblock.hpp
    @author  Jules Bergmann
    @date    02/11/2005
    @brief   VSIPL++ Library: Plain block class.

    Plain block class. similar to Dense, but does not provide
    admit/release and may not implement Direct_data (depending on
    PLAINBLOCK_ENABLE_DIRECT_DATA define).
*/

#ifndef VSIP_PLAINBLOCK_HPP
#define VSIP_PLAINBLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/
#include <stdexcept>
#include <string>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/extdata.hpp>
#include <vsip/core/block_traits.hpp>



/***********************************************************************
  Macros
***********************************************************************/

/// Control whether Direct_data access is supported.
#ifndef PLAINBLOCK_ENABLE_DIRECT_DATA
#  define PLAINBLOCK_ENABLE_DIRECT_DATA 0
#endif

/// Control whether impl_ref() is supported.
#ifndef PLAINBLOCK_ENABLE_IMPL_REF
#  define PLAINBLOCK_ENABLE_IMPL_REF 0
#endif



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// Plain block, as defined in standard [view.dense].

/// "A Plain block is a modifiable, allocatable 1-dimensional block
/// or 1,x-dimensional block, for a fixed x, that explicitly stores
/// one value for each Index in its domain."

template <dimension_type Dim   = 1,
	  typename    T        = VSIP_DEFAULT_VALUE_TYPE,
	  typename    Order    = typename impl::Row_major<Dim>::type,
	  typename    Map      = Local_map>
class Plain_block;



/// Partial specialization of Plain_block class template for 1-dimension.

/// Note: This declaration is incomplete.  The following items
///       required by the spec are not declared (and not defined)
///         - User defined storage, including user storage
///           constructors, admit, release, find, and rebind.
///
///       The following items required by the spec are declared
///       but not implemented:
///         - The user_storage() and admitted() accessors.
///

template <typename    T,
	  typename    Order,
	  typename    DenseMap>
class Plain_block<1, T, Order, DenseMap>
  : public impl::Ref_count<Plain_block<1, T, Order, DenseMap> >
{
  // Compile-time values and types.
public:
  static dimension_type const dim = 1;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef Order    order_type;
  typedef DenseMap map_type;

  // Enable Direct_data access to data.
  template <typename, typename, typename>
  friend class impl::data_access::Low_level_data_access;

  // Implementation types.
public:
  typedef impl::Layout<dim, order_type, impl::Stride_unit_dense, impl::Cmplx_inter_fmt>
		layout_type;
  typedef impl::Applied_layout<layout_type>
		applied_layout_type;
  typedef impl::Allocated_storage<typename layout_type::complex_type, T> storage_type;

  // Constructors and destructor.
public:
  Plain_block(Domain<1> const& dom, DenseMap const& = DenseMap())
    VSIP_THROW((std::bad_alloc));

  Plain_block(Domain<1> const& dom, T value, DenseMap const& = DenseMap())
    VSIP_THROW((std::bad_alloc));

  ~Plain_block() VSIP_NOTHROW;

  // Data accessors.
public:
  T get(index_type idx) const VSIP_NOTHROW;
  void put(index_type idx, T val) VSIP_NOTHROW;

#if PLAINBLOCK_ENABLE_IMPL_REF
  reference_type       impl_ref(index_type idx) VSIP_NOTHROW;
  const_reference_type impl_ref(index_type idx) const VSIP_NOTHROW;
#endif

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW;
  length_type size(dimension_type Dim, dimension_type d) const VSIP_NOTHROW;
  DenseMap const& map() const VSIP_NOTHROW { return map_;}

  // Support Direct_data interface.
public:
  typedef typename storage_type::type       data_type;
  typedef typename storage_type::const_type const_data_type;

  data_type       impl_data()       VSIP_NOTHROW { return storage_.data(); }
  const_data_type impl_data() const VSIP_NOTHROW { return storage_.data(); }
  stride_type impl_stride(dimension_type Dim, dimension_type d)
    const VSIP_NOTHROW;

  // Hidden copy constructor and assignment.
private:
  Plain_block(Plain_block const&);
  Plain_block& operator=(Plain_block const&);

  // Member Data
private:
  applied_layout_type layout_;
  storage_type        storage_;
  map_type            map_;
};



/// Partial specialization of Plain_block class template for 1,2-dimension.

/// Note: This declaration is incomplete.  The following items
///       required by the spec are not declared (and not defined)
///         - User defined storage, including user storage
///           constructors, admit, release, find, and rebind.
///
///       The following items required by the spec are declared
///       but not implemented:
///         - The user_storage() and admitted() accessors.
///

template <typename    T,
	  typename    Order,
	  typename    DenseMap>
class Plain_block<2, T, Order, DenseMap>
  : public impl::Ref_count<Plain_block<2, T, Order, DenseMap> >
{
  // Compile-time values and types.
public:
  static dimension_type const dim = 2;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef Order    order_type;
  typedef DenseMap map_type;

  // Implementation types.
public:
  typedef impl::Layout<dim, order_type, impl::Stride_unit_dense, impl::Cmplx_inter_fmt>
		layout_type;
  typedef impl::Applied_layout<layout_type>
		applied_layout_type;
  typedef impl::Allocated_storage<typename layout_type::complex_type, T> storage_type;

  // Constructors and destructor.
public:
  Plain_block(Domain<2> const& dom, DenseMap const& = DenseMap())
    VSIP_THROW((std::bad_alloc));

  Plain_block(Domain<2> const& dom, T value, DenseMap const& = DenseMap())
    VSIP_THROW((std::bad_alloc));

  ~Plain_block() VSIP_NOTHROW;

  // Data Accessors.
public:
  T get(index_type idx) const VSIP_NOTHROW;
  void put(index_type idx, T val) VSIP_NOTHROW;

  T get(index_type idx0, index_type idx1) const VSIP_NOTHROW;
  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW;

#if PLAINBLOCK_ENABLE_IMPL_REF
  reference_type       impl_ref(index_type idx) VSIP_NOTHROW;
  const_reference_type impl_ref(index_type idx) const VSIP_NOTHROW;
  reference_type       impl_ref(index_type idx0, index_type idx1) VSIP_NOTHROW;
  const_reference_type impl_ref(index_type idx0, index_type idx1) const VSIP_NOTHROW;
#endif

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW;
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW;
  DenseMap const& map() const VSIP_NOTHROW { return map_;}

  // Support Direct_data interface.
public:
  typedef typename storage_type::type       data_type;
  typedef typename storage_type::const_type const_data_type;

  data_type       impl_data()       VSIP_NOTHROW { return storage_.data(); }
  const_data_type impl_data() const VSIP_NOTHROW { return storage_.data(); }
  stride_type impl_stride(dimension_type Dim, dimension_type d)
    const VSIP_NOTHROW;

  // Hidden copy constructor and assignment.
private:
  Plain_block(Plain_block const&);
  Plain_block& operator=(Plain_block const&);

  // Member data.
private:
  applied_layout_type layout_;
  storage_type        storage_;
  map_type            map_;
};



namespace impl
{

template <dimension_type Dim,
	  typename       T,
	  typename       Order,
	  typename       Map>
struct Block_layout<Plain_block<Dim, T, Order, Map> >
{
  static dimension_type const dim = Dim;

  typedef Direct_access_tag access_type;
  typedef Order           order_type;
  typedef Stride_unit_dense pack_type;
  typedef Cmplx_inter_fmt   complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> layout_type;
};

#if PLAINBLOCK_ENABLE_IMPL_REF
template <dimension_type Dim,
          typename       T,
          typename       Order,
          typename       Map>
struct Lvalue_factory_type<Plain_block<Dim, T, Order, Map> >
{
  typedef True_lvalue_factory<Plain_block<Dim, T, Order, Map> > type;
  template <typename OtherBlock>
  struct Rebind {
    typedef True_lvalue_factory<OtherBlock> type;
  };
};
#endif

} // namespace vsip::impl



/***********************************************************************
  Definitions
***********************************************************************/

// Plain_block<1, T, Order, Map>

template <typename    T,
	  typename    Order,
	  typename    Map>
inline
Plain_block<1, T, Order, Map>::Plain_block(Domain<1> const& dom, Map const& map)
  VSIP_THROW((std::bad_alloc))
  : layout_    (dom),
    storage_   (layout_.total_size()),
    map_       (map)
{
}



template <typename    T,
	  typename    Order,
	  typename    Map>
inline
Plain_block<1, T, Order, Map>::Plain_block(Domain<1> const& dom, T val, Map const& map)
  VSIP_THROW((std::bad_alloc))
  : layout_    (dom),
    storage_   (layout_.total_size(), val),
    map_       (map)
{
}



template <typename    T,
	  typename    Order,
	  typename    Map>
inline
Plain_block<1, T, Order, Map>::~Plain_block()
  VSIP_NOTHROW
{
  storage_.deallocate(layout_.total_size());
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
T
Plain_block<1, T, Order, Map>::get(
  index_type idx)
  const VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.get(idx);
}



template <typename    T,
	  typename    Order,
	  typename    Map>
inline
void
Plain_block<1, T, Order, Map>::put(
  index_type idx,
  T       val)
  VSIP_NOTHROW
{
  assert(idx < size());
  storage_.put(idx, val);
}


#if PLAINBLOCK_ENABLE_IMPL_REF
template <typename    T,
	  typename    Order,
	  typename    Map >
inline
typename Plain_block<1, T, Order, Map>::reference_type
Plain_block<1, T, Order, Map>::impl_ref(index_type idx) VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.ref(idx);
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
typename Plain_block<1, T, Order, Map>::const_reference_type
Plain_block<1, T, Order, Map>::impl_ref(index_type idx) const VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.ref(idx);
}
#endif


/// Return the total size of the block.
template <typename    T,
	  typename    Order,
	  typename    Map>
inline
length_type
Plain_block<1, T, Order, Map>::size() const VSIP_NOTHROW
{
  return layout_.size(0);
}



/// Return the size of the block in a specific dimension.

/// Requires:
///   BLOCK_DIM selects which block-dimensionality (BLOCK_DIM == 1).
///   DIM is the dimension whose length to return (0 <= DIM < BLOCK_DIM).
/// Returns:
///   The size of dimension DIM.
template <typename    T,
	  typename    Order,
	  typename    Map>
inline
length_type
Plain_block<1, T, Order, Map>::size(
  dimension_type block_dim,
  dimension_type dim)
  const VSIP_NOTHROW
{
  assert(block_dim == 1);
  assert(dim == 0);
  return layout_.size(0);
}



// Requires:
//   DIM is a valid dimensionality supported by block (DIM must be 1).
//   D is a dimension, less than DIM (D must be 0).
// Returns
//   The stride in dimension D, for dimensionality DIM.

template <typename    T,
	  typename    Order,
	  typename    Map>
inline
stride_type
Plain_block<1, T, Order, Map>::impl_stride(dimension_type Dim, dimension_type d)
  const VSIP_NOTHROW
{
  assert(Dim == dim && d == 0);
  return 1;
}



/**********************************************************************/
// Plain_block<2, T, Order, Map>

/// Construct a 1,2-dimensional Plain_block block.
template <typename    T,
	  typename    Order,
	  typename    Map>
inline
Plain_block<2, T, Order, Map>::Plain_block(Domain<2> const& dom, Map const& map)
  VSIP_THROW((std::bad_alloc))
  : layout_ (dom[0].size(), dom[1].size()),
    storage_(layout_.total_size()),
    map_    (map)
{
}



/// Construct a 1,2-dimensional Plain_block block and initialize data.
template <typename    T,
	  typename    Order,
	  typename    Map>
inline
Plain_block<2, T, Order, Map>::Plain_block(Domain<2> const& dom, T val, Map const& map)
  VSIP_THROW((std::bad_alloc))
  : layout_ (dom[0].size(), dom[1].size()),
    storage_(layout_.total_size(), val),
    map_    (map)
{
}



template <typename    T,
	  typename    Order,
	  typename    Map>
inline
Plain_block<2, T, Order, Map>::~Plain_block()
  VSIP_NOTHROW
{
  storage_.deallocate(layout_.total_size());
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
T
Plain_block<2, T, Order, Map>::get(
  index_type idx)
  const VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.get(idx);
}



template <typename    T,
	  typename    Order,
	  typename    Map>
inline
void
Plain_block<2, T, Order, Map>::put(
  index_type idx,
  T       val)
  VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.put(idx, val);
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
T
Plain_block<2, T, Order, Map>::get(
  index_type idx0,
  index_type idx1)
  const VSIP_NOTHROW
{
  assert((idx0 < layout_.size(0)) && (idx1 < layout_.size(1)));
  return storage_.get(layout_.index(idx0, idx1));
}



template <typename    T,
	  typename    Order,
	  typename    Map>
inline
void
Plain_block<2, T, Order, Map>::put(
  index_type idx0,
  index_type idx1,
  T       val)
  VSIP_NOTHROW
{
  assert((idx0 < layout_.size(0)) && (idx1 < layout_.size(1)));
  storage_.put(layout_.index(idx0, idx1), val);
}


#if PLAINBLOCK_ENABLE_IMPL_REF
template <typename    T,
	  typename    Order,
	  typename    Map >
inline
typename Plain_block<2, T, Order, Map>::reference_type
Plain_block<2, T, Order, Map>::impl_ref(index_type idx) VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.ref(idx);
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
typename Plain_block<2, T, Order, Map>::const_reference_type
Plain_block<2, T, Order, Map>::impl_ref(index_type idx) const VSIP_NOTHROW
{
  assert(idx < size());
  return storage_.ref(idx);
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
typename Plain_block<2, T, Order, Map>::reference_type
Plain_block<2, T, Order, Map>::impl_ref(
  index_type idx0,
  index_type idx1) VSIP_NOTHROW
{
  assert((idx0 < layout_.size(0)) && (idx1 < layout_.size(1)));
  return storage_.ref(layout_.index(idx0, idx1));
}



template <typename    T,
	  typename    Order,
	  typename    Map >
inline
typename Plain_block<2, T, Order, Map>::const_reference_type
Plain_block<2, T, Order, Map>::impl_ref(
  index_type idx0,
  index_type idx1) const VSIP_NOTHROW
{
  assert((idx0 < layout_.size(0)) && (idx1 < layout_.size(1)));
  return storage_.ref(layout_.index(idx0, idx1));
}
#endif


/// Return the total size of the block.
template <typename    T,
	  typename    Order,
	  typename    Map>
inline
length_type
Plain_block<2, T, Order, Map>::size() const VSIP_NOTHROW
{
  return layout_.size(0) * layout_.size(1);
}



/// Return the size of the block in a specific dimension.

/// Requires:
///   BLOCK_DIM selects which block-dimensionality (BLOCK_DIM <= 2).
///   DIM is the dimension whose length to return (0 <= DIM < BLOCK_DIM).
/// Returns:
///   The size of dimension DIM.
template <typename    T,
	  typename    Order,
	  typename    Map>
inline
length_type
Plain_block<2, T, Order, Map>::size(
  dimension_type block_dim,
  dimension_type dim)
  const VSIP_NOTHROW
{
  assert((block_dim == 1 || block_dim == 2) && (dim < block_dim));

  if (block_dim == 1)
    return size();
  else
    return (dim == 0) ? layout_.size(0) : layout_.size(1);
}



// Requires:
//   DIM is a valid dimensionality supported by block (DIM == 1 or 2)
//   D is a dimension, less than DIM.
// Returns
//   The stride in dimension D, for dimensionality DIM.

template <typename    T,
	  typename    Order,
	  typename    Map>
inline
stride_type
Plain_block<2, T, Order, Map>::impl_stride(dimension_type Dim, dimension_type d)
  const VSIP_NOTHROW
{
  assert(Dim == 1 || Dim == dim);
  assert(d < Dim);

  if (Dim == 1)
    return 1;
  else
    return layout_.stride(d);
}



} // namespace vsip

#endif // VSIP_DENSE_HPP
