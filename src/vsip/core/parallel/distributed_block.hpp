/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

#ifndef vsip_core_parallel_distributed_block_hpp_
#define vsip_core_parallel_distributed_block_hpp_

#include <vsip/support.hpp>
#if VSIP_IMPL_PAR_SERVICE == 2
# include <vsip/opt/pas/block.hpp>
#else
# include <vsip/core/mpi/distributed_block.hpp>
#endif

namespace vsip
{
namespace impl
{
#if VSIP_IMPL_PAR_SERVICE == 2
using pas::Distributed_block;
#else
using mpi::Distributed_block;
#endif

template <typename B, typename M>
struct Distributed_local_block<Distributed_block<B, M> >
{
  typedef typename Distributed_block<B, M>::local_block_type type;
  typedef typename Distributed_block<B, M>::proxy_local_block_type proxy_type;
};

template <typename B, typename M>
struct Is_simple_distributed_block<Distributed_block<B, M> >
{
  static bool const value = true;
};

#if VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES==0

template <typename CombineT, typename B, typename M>
struct Combine_return_type<CombineT, Distributed_block<B, M> >
{
  typedef Distributed_block<B, M> block_type;
  typedef typename CombineT::template return_type<block_type>::type type;
  typedef typename CombineT::template tree_type<block_type>::type tree_type;
};

template <typename CombineT, typename B, typename M>
typename Combine_return_type<CombineT, Distributed_block<B, M> >::type
apply_combine(CombineT const &combine, Distributed_block<B, M> const &block)
{
  return combine.apply(block);
}

#endif

/// Return the local block for a given subblock.

#if 0
// For now, leave this undefined to catch unhandled distributed cases at
// compile-time.
template <typename B>
typename Distributed_local_block<B>::type&
get_local_block(B const &)
{
  // In general case, we should assume block is not distributed and
  // just return it.
  //
  // For now, through exception to catch unhandled distributed cases.
  VSIP_IMPL_THROW(impl::unimplemented("get_local_block()"));
}
#endif

template <typename B, typename M>
B &get_local_block(Distributed_block<B, M> const &block)
{
  return block.get_local_block();
}

#if 0
// For now, leave this undefined to catch unhandled distributed cases at
// compile-time.
template <typename B>
void
assert_local(B const & /*block*/, index_type sb)
{
  // In general case, we should assume block is not distributed and
  // just return it.
  //
  // For now, through exception to catch unhandled distributed cases.
  VSIP_IMPL_THROW(impl::unimplemented("assert_local()"));
}
#endif

template <typename B, typename M>
void assert_local(Distributed_block<B, M> const &block, index_type sb)
{
  block.assert_local(sb);
}

} // namespace vsip::impl

template <typename B, typename M>
struct get_block_layout<impl::Distributed_block<B, M> > : get_block_layout<B> {};

template <typename B, typename M>
struct supports_dda<impl::Distributed_block<B, M> > : supports_dda<B> {};

} // namespace vsip


#endif
