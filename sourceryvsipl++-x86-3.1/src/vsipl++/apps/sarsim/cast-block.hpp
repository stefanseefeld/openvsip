/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    cast-block.hpp
    @author  Jules Bergmann
    @date    06/15/2005
    @brief   VSIPL++ Library: Cast block class.

*/

#ifndef VSIP_IMPL_CAST_BLOCK_HPP
#define VSIP_IMPL_CAST_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/noncopyable.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

template <typename T,
	  typename Block>
class Cast_block : public impl::Non_assignable
{
  // Compile-time values and types.
public:
  static dimension_type const dim = Block::dim;

  typedef T        value_type;
  typedef T&       reference_type;
  typedef T const& const_reference_type;

  typedef typename Block::map_type map_type;

  // Constructors and destructor.
public:
  Cast_block(Block& block) VSIP_NOTHROW : block_(&block) {}
  Cast_block(Cast_block const& b) : block_(&*b.block_) {}
  ~Cast_block() VSIP_NOTHROW {}

  // Accessors.
public:
  length_type size() const VSIP_NOTHROW
    { return block_->size();}

  length_type size(dimension_type block_d, dimension_type d) const VSIP_NOTHROW
    { return block_->size(block_d, d);}

  // These are noops as Cast_block is held by-value.
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const { return block_->map();}

  // Data accessors
public:
  value_type get(index_type i) const VSIP_NOTHROW
    { return static_cast<T>(this->block_->get(i)); }

  value_type get(index_type i, index_type j) const VSIP_NOTHROW
    { return static_cast<T>(this->block_->get(i, j)); }

  value_type get(index_type i, index_type j, index_type k) const VSIP_NOTHROW
    { return static_cast<T>(this->block_->get(i, j, k)); }


  // Member data.
private:
  typename View_block_storage<Block>::type block_;
};

// Store Cast_blocks by-value.
template <typename T,
	  typename Block>
struct View_block_storage<Cast_block<T, Block> >
  : By_value_block_storage<Cast_block<T, Block> >
{};


template <typename                            T,
	  template <typename, typename> class V,
	  typename                            T1,
	  typename                            Block1>
struct Cast_view
{
  typedef V<T, Cast_block<T, Block1> > view_type;

  static view_type cast(V<T1, Block1> const& v)
  {
    Cast_block<T, Block1> block(v.block());
    return view_type(block);
  }
};



/// Specialization to avoid unnecessary cast when T == T1.

template <typename                            T,
	  template <typename, typename> class V,
	  typename                            Block1>
struct Cast_view<T, V, T, Block1>
{
  typedef V<T, Block1> view_type;

  static view_type cast(V<T, Block1> const& v)
  {
    return v;
  }
};



/***********************************************************************
  Definitions
***********************************************************************/

template <typename                            T,
	  template <typename, typename> class V,
	  typename                            T1,
	  typename                            Block1>
typename Cast_view<T, V, T1, Block1>::view_type
cast_view(V<T1, Block1> const& view)
{
  return Cast_view<T, V, T1, Block1>::cast(view);
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_CAST_BLOCK_HPP
	     
