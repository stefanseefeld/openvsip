//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_get_local_view_hpp_
#define ovxx_parallel_get_local_view_hpp_

#include <ovxx/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/domain_utils.hpp>

namespace ovxx
{
namespace parallel
{

/// Get a local view of a subblock.

template <template <typename, typename> class V,
	  typename T, typename B, typename M = typename B::map_type>
struct get_local_view
{
  static V<T, typename distributed_local_block<B>::type> exec(V<T, B> v)
  {
    typedef typename distributed_local_block<B>::type block_type;
    typedef typename block_traits<block_type>::plain_type storage_type;

    storage_type block = get_local_block(v.block());
    return V<T, block_type>(block);
  }
};

template <template <typename, typename> class V,
	  typename T, typename B>
struct get_local_view<V, T, B, Local_map>
{
  static V<T, typename distributed_local_block<B>::type>
  exec(V<T, B> v)
  {
    typedef typename distributed_local_block<B>::type block_type;
    assert((is_same<B, block_type>::value));
    return v;
  }
};
	  
} // namespace ovxx::parallel

template <template <typename, typename> class V, typename T, typename B>
V<T, typename distributed_local_block<B>::type>
get_local_view(V<T, B> v)
{
  return parallel::get_local_view<V, T, B>::exec(v);
}

template <template <typename, typename> class V, typename T, typename B>
void
view_assert_local(V<T, B> v, index_type i)
{
  assert_local(v.block(), i);
}

} // namespace ovxx

#endif
