/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/opt/get_local_view.hpp
    @author  Jules Bergmann
    @date    2005-03-22
    @brief   VSIPL++ Library: Get_local_view function & helper class.

*/

#ifndef VSIP_OPT_GET_LOCAL_VIEW_HPP
#define VSIP_OPT_GET_LOCAL_VIEW_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/domain_utils.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

/// Get a local view of a subblock.

template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block,
	  typename                            MapT = typename Block::map_type>
struct Get_local_view_class
{
  static
  View<T, typename Distributed_local_block<Block>::type>
  exec(
    View<T, Block> v)
  {
    typedef typename Distributed_local_block<Block>::type block_t;
    typedef typename View_block_storage<block_t>::type::equiv_type storage_t;

    storage_t blk = get_local_block(v.block());
    return View<T, block_t>(blk);
  }
};

template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block>
struct Get_local_view_class<View, T, Block, Local_map>
{
  static
  View<T, typename Distributed_local_block<Block>::type>
  exec(
    View<T, Block> v)
  {
    typedef typename Distributed_local_block<Block>::type block_t;
    assert((Type_equal<Block, block_t>::value));
    return v;
  }
};
	  


template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block>
View<T, typename Distributed_local_block<Block>::type>
get_local_view(
  View<T, Block> v)
{
  return Get_local_view_class<View, T, Block>::exec(v);
}



template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block>
void
view_assert_local(
  View<T, Block> v,
  index_type     sb)
{
  assert_local(v.block(), sb);
}



} // namespace vsip::impl
} // namespace vsip


#endif // VSIP_IMPL_GET_LOCAL_VIEW_HPP
