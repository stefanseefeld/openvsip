/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */
#ifndef vsip_core_assign_local_hpp_
#define vsip_core_assign_local_hpp_

#include <vsip/support.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip
{
namespace impl
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
	  bool IsLocal1 = Is_local_map<typename Block1::map_type>::value,
	  bool IsLocal2 = Is_local_map<typename Block2::map_type>::value>
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

/// Assign between local and distributed views.
template <typename LHS, typename RHS>
void assign_local(LHS lhs, RHS rhs,
		  typename enable_if_c<Is_view_type<LHS>::value &&
		                       Is_view_type<RHS>::value>::type * = 0)
{
  Assign_local<typename LHS::block_type, typename RHS::block_type>
    ::exec(lhs.block(), rhs.block());
}

/// Assign between local and distributed views.
template <bool Predicate, typename LHS, typename RHS>
void assign_local_if(LHS lhs, RHS rhs,
	 	     typename enable_if_c<Is_view_type<LHS>::value &&
		                          Is_view_type<RHS>::value>::type * = 0)
{
  Assign_local_if<Predicate, typename LHS::block_type, typename RHS::block_type>
    ::exec(lhs.block(), rhs.block());
}


/// Assign between local and distributed blocks.
template <typename LHS, typename RHS>
void assign_local(LHS &lhs, RHS const &rhs,
		  typename enable_if_c<!Is_view_type<LHS>::value ||
		                       !Is_view_type<RHS>::value>::type * = 0)
{
  Assign_local<LHS, RHS>::exec(lhs, rhs);
}

/// Assign between local and distributed blocks.
template <bool Predicate, typename LHS, typename RHS>
void assign_local_if(LHS &lhs, RHS const &rhs,
		     typename enable_if_c<!Is_view_type<LHS>::value ||
		                          !Is_view_type<RHS>::value>::type * = 0)
{
  Assign_local_if<Predicate, LHS, RHS>::exec(lhs, rhs);
}

} // namespace vsip::impl
} // namespace vsip

#endif
