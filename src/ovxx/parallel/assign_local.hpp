//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_assign_local_hpp_
#define ovxx_parallel_assign_local_hpp_

#include <ovxx/support.hpp>
#include <ovxx/parallel/service.hpp>
#include <ovxx/ct_assert.hpp>

namespace ovxx
{
namespace parallel
{
namespace detail
{

/// Used for working with distributed data by replicating a copy locally
/// to each processor.
///
///   - Assign_local() transfers data between local and distributed views.
///   - Working_view_holder creates a local working view of an argument,
///     either replicating a distributed view to a local copy, or aliasing
///     a local view.
template <typename Block1,
	  typename Block2,
	  bool IsLocal1 = is_local_map<typename Block1::map_type>::value,
	  bool IsLocal2 = is_local_map<typename Block2::map_type>::value>
struct Assign_local;

// Local assignment
template <typename LHS, typename RHS>
struct Assign_local<LHS, RHS, true, true>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    assign<LHS::dim>(lhs, rhs);
  }
};

// Distributed to local assignment.
// Copy rhs into a replicated block, then copy that to local.
template <typename LHS, typename RHS>
struct Assign_local<LHS, RHS, true, false>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef typename LHS::value_type value_type;
    typedef Replicated_map<LHS::dim> map_type;
    typedef typename get_block_layout<LHS>::order_type order_type;
    typedef Dense<LHS::dim, value_type, order_type, map_type> replicated_block_type;

    replicated_block_type repl(block_domain<LHS::dim>(rhs));
    assign<LHS::dim>(repl, rhs);
    assign<LHS::dim>(lhs, repl.get_local_block());
  }
};

// Local to distributed assignment.
// Copy rhs into a replicated block, then copy that to global.
template <typename LHS, typename RHS>
struct Assign_local<LHS, RHS, false, true>
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef typename LHS::value_type value_type;
    typedef Replicated_map<LHS::dim> map_type;
    typedef typename get_block_layout<LHS>::order_type order_type;
    typedef Dense<LHS::dim, value_type, order_type, map_type> replicated_block_type;

    replicated_block_type repl(block_domain<LHS::dim>(rhs));
    assign<LHS::dim>(repl.get_local_block(), rhs);
    assign<LHS::dim>(lhs, repl);
  }
};

/// Guarded Assign_local
template <bool Predicate, typename LHS, typename RHS>
struct Assign_local_if
{
  static void exec(LHS &lhs, RHS const &rhs)
  {
    Assign_local<LHS, RHS>::exec(lhs, rhs);
  }
};

template <typename LHS, typename RHS>
struct Assign_local_if<false, LHS, RHS>
{
  static void exec(LHS &, RHS const &) {}
};

} // namespace ovxx::parallel::detail

/// Assign between local and distributed views.
template <typename LHS, typename RHS>
void assign_local(LHS lhs, RHS rhs,
		  typename enable_if<is_view_type<LHS>::value &&
		                     is_view_type<RHS>::value>::type * = 0)
{
  detail::Assign_local<typename LHS::block_type, typename RHS::block_type>
    ::exec(lhs.block(), rhs.block());
}

/// Assign between local and distributed views.
template <bool P, typename LHS, typename RHS>
void assign_local_if(LHS lhs, RHS rhs,
	 	     typename enable_if<is_view_type<LHS>::value &&
		                        is_view_type<RHS>::value>::type * = 0)
{
  detail::Assign_local_if<P, typename LHS::block_type, typename RHS::block_type>
    ::exec(lhs.block(), rhs.block());
}


/// Assign between local and distributed blocks.
template <typename LHS, typename RHS>
void assign_local(LHS &lhs, RHS const &rhs,
		  typename enable_if<!is_view_type<LHS>::value ||
		                     !is_view_type<RHS>::value>::type * = 0)
{
  detail::Assign_local<LHS, RHS>::exec(lhs, rhs);
}

/// Assign between local and distributed blocks.
template <bool P, typename LHS, typename RHS>
void assign_local_if(LHS &lhs, RHS const &rhs,
		     typename enable_if<!is_view_type<LHS>::value ||
		                        !is_view_type<RHS>::value>::type * = 0)
{
  detail::Assign_local_if<P, LHS, RHS>::exec(lhs, rhs);
}

} // namespace ovxx::parallel
} // namespace ovxx

#endif
