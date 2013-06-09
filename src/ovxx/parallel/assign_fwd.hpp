//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_assign_fwd_hpp_
#define ovxx_parallel_assign_fwd_hpp_

#include <vsip/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/parallel/map_traits.hpp>

namespace ovxx
{
namespace parallel
{

struct Chained_assign;
struct Blkvec_assign;

// Parallel assignment.
template <dimension_type D, typename LHS, typename RHS, typename ImplTag>
class Assignment;

template <dimension_type D, typename LHS, typename RHS, bool EarlyBinding>
struct choose_par_assign_impl
{
  typedef typename LHS::map_type map1_type;
  typedef typename RHS::map_type map2_type;

  static int const  is_blkvec = (D == 1) &&
    is_block_dist<0, map1_type>::value &&
    is_block_dist<0, map2_type>::value;

  typedef typename conditional<is_blkvec, Blkvec_assign, Chained_assign>::type type;
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
