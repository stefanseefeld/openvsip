//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_vmmul_hpp_
#define ovxx_expr_vmmul_hpp_

#include <ovxx/block_traits.hpp>
#include <vsip/impl/promotion.hpp>
#include <vsip/impl/map_fwd.hpp>
#include <ovxx/parallel/map_traits.hpp>

namespace ovxx
{
namespace expr
{

/// Expression template block for vector-matrix multiply.
/// 
/// Template parameters:   
///   :D: a dimension of vector (0 or 1)
///   :Block0: a 1-Dim Block.
///   :Block1: a 2-Dim Block.
template <dimension_type D, typename Block0, typename Block1>
class Vmmul : ovxx::detail::nonassignable
{
public:
  static dimension_type const dim = 2;

  typedef typename Block0::value_type value0_type;
  typedef typename Block1::value_type value1_type;

  typedef typename vsip::Promotion<value0_type, value1_type>::type value_type;

  typedef value_type&               reference_type;
  typedef value_type const&         const_reference_type;
  typedef typename Block1::map_type map_type;

  Vmmul(Block0 const& vblk, Block1 const& mblk) : vblk_(vblk), mblk_(mblk) {}

  length_type size() const VSIP_NOTHROW { return mblk_.size(); }
  length_type size(dimension_type Dim, dimension_type d) const VSIP_NOTHROW
  { return mblk_.size(Dim, d); }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW { return mblk_.map();}

  value_type get(index_type i, index_type j) const
  {
    if (D == 0)
      return vblk_.get(j) * mblk_.get(i, j);
    else
      return vblk_.get(i) * mblk_.get(i, j);
  }

  Block0 const& get_vblk() const VSIP_NOTHROW { return vblk_; }
  Block1 const& get_mblk() const VSIP_NOTHROW { return mblk_; }

  // copy-constructor: default is OK.

private:
  typename block_traits<Block0>::expr_type vblk_;
  typename block_traits<Block1>::expr_type mblk_;
};

} // namespace ovxx::expr

template <dimension_type D, typename V, typename M>
struct is_expr_block<expr::Vmmul<D, V, M> >
{ static bool const value = true;};

template <dimension_type D, typename V, typename M>
struct block_traits<expr::Vmmul<D, V, M> const>
  : by_value_traits<expr::Vmmul<D, V, M> const>
{};

template <dimension_type D, typename V, typename M>
struct distributed_local_block<expr::Vmmul<D, V, M> const>
{
  typedef expr::Vmmul<D,
		      typename distributed_local_block<V>::type,
		      typename distributed_local_block<M>::type>
    const type;
  typedef expr::Vmmul<D,
		      typename distributed_local_block<V>::proxy_type,
		      typename distributed_local_block<M>::proxy_type>
    const proxy_type;
};

namespace detail 
{
  
template <dimension_type D, typename V, typename M>
expr::Vmmul<D, 
	    typename distributed_local_block<V>::type,
	    typename distributed_local_block<M>::type>
get_local_block(expr::Vmmul<D, V, M> const &block)
{
  typedef expr::Vmmul<D,
		      typename distributed_local_block<V>::type,
		      typename distributed_local_block<M>::type>
    block_type;

  return block_type(get_local_block(block.get_vblk()),
		    get_local_block(block.get_mblk()));
}

} // namespace ovxx::detail

namespace parallel
{
// Check vmmul parallel support conditions
//
// vector-matrix multiply works with the following mappings:
// case 0:
//  - All data mapped locally (Local_map) (*)
// case 1:
//  - vector data mapped global
//    matrix data mapped without distribution only vector direction
// case 2:
//  - vector data mapped distributed,
//    matrix data mapped with same distribution along vector direction,
//       and no distribution perpendicular to vector.
//  - vector and matrix mapped to single, single processor
//

template <dimension_type D,
	  typename       M,
	  dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
bool has_same_map(M const &map, expr::Vmmul<VecDim, Block0, Block1> const &block)
{
  using namespace parallel;
  return 
    // Case 1: vector is replicated
    (has_same_map<1>(Replicated_map<1>(), block.get_vblk()) &&
     map.num_subblocks(1-VecDim) == 1 &&
     has_same_map<D>(map, block.get_mblk())) ||

    // Case 2:
    (map.num_subblocks(VecDim) == 1 &&
     has_same_map<1>(map_project_1<VecDim, M>::project(map, 0),
		     block.get_vblk()) &&
     has_same_map<D>(map, block.get_mblk()));
}

template <dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct is_reorg_ok<expr::Vmmul<VecDim, Block0, Block1> const>
{
  static bool const value = false;
};
} // namespace ovxx::parallel
} // namespace ovxx

#endif
