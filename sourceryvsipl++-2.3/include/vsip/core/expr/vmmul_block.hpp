/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/expr/vmmul_block.hpp
    @author  Jules Bergmann
    @date    2007-02-02
    @brief   VSIPL++ Library: Expression block for vector-matrix multiply

*/

#ifndef VSIP_CORE_EXPR_VMMUL_BLOCK_HPP
#define VSIP_CORE_EXPR_VMMUL_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/block_traits.hpp>
#include <vsip/core/promote.hpp>
#include <vsip/core/map_fwd.hpp>
#include <vsip/core/parallel/map_traits.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
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
class Vmmul : public impl::Non_assignable
{
public:
  static dimension_type const dim = 2;

  typedef typename Block0::value_type value0_type;
  typedef typename Block1::value_type value1_type;

  typedef typename Promotion<value0_type, value1_type>::type value_type;

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
  typename impl::View_block_storage<Block0>::expr_type vblk_;
  typename impl::View_block_storage<Block1>::expr_type mblk_;
};
} // namespace vsip_csl::expr
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

/// Specialize traits for Vmmul_expr_block.

template <dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct Is_expr_block<expr::Vmmul<VecDim, Block0, Block1> >
{ static bool const value = true; };

template <dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct View_block_storage<expr::Vmmul<VecDim, Block0, Block1> const>
  : By_value_block_storage<expr::Vmmul<VecDim, Block0, Block1> const>
{};

template <dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct Distributed_local_block<expr::Vmmul<VecDim, Block0, Block1> const>
{
  typedef expr::Vmmul<VecDim,
		      typename Distributed_local_block<Block0>::type,
		      typename Distributed_local_block<Block1>::type>
		const type;
  typedef expr::Vmmul<VecDim,
		      typename Distributed_local_block<Block0>::proxy_type,
		      typename Distributed_local_block<Block1>::proxy_type>
		const proxy_type;
};



template <typename       CombineT,
	  dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct Combine_return_type<CombineT,
                           expr::Vmmul<VecDim, Block0, Block1> const>
{
  typedef expr::Vmmul<VecDim,
    typename Combine_return_type<CombineT, Block0>::tree_type,
    typename Combine_return_type<CombineT, Block1>::tree_type>
		const tree_type;
  typedef tree_type type;
};



template <typename       CombineT,
	  dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct Combine_return_type<CombineT,
                           expr::Vmmul<VecDim, Block0, Block1> >
{
  typedef expr::Vmmul<VecDim,
    typename Combine_return_type<CombineT, Block0>::tree_type,
    typename Combine_return_type<CombineT, Block1>::tree_type>
		const tree_type;
  typedef tree_type type;
};


  
template <dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
expr::Vmmul<VecDim, 
		 typename Distributed_local_block<Block0>::type,
		 typename Distributed_local_block<Block1>::type>
get_local_block(expr::Vmmul<VecDim, Block0, Block1> const& block)
{
  typedef expr::Vmmul<VecDim,
    typename Distributed_local_block<Block0>::type,
    typename Distributed_local_block<Block1>::type>
		block_type;

  return block_type(get_local_block(block.get_vblk()),
		    get_local_block(block.get_mblk()));
}



template <typename       CombineT,
	  dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
typename Combine_return_type<CombineT,
			     expr::Vmmul<VecDim, Block0, Block1> const>
		::type
apply_combine(CombineT const &combine, expr::Vmmul<VecDim, Block0, Block1> const& block)
{
  typedef typename Combine_return_type<
    CombineT,
      expr::Vmmul<VecDim, Block0, Block1> const>::type
		block_type;

  return block_type(apply_combine(combine, block.get_vblk()),
		    apply_combine(combine, block.get_mblk()));
}



template <typename       VisitorT,
	  dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
void
apply_leaf(VisitorT const &visitor,
	   expr::Vmmul<VecDim, Block0, Block1> const& block)
{
  apply_leaf(visitor, block.get_vblk());
  apply_leaf(visitor, block.get_mblk());
}



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

template <dimension_type MapDim,
	  typename       MapT,
	  dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct Is_par_same_map<MapDim, MapT,
                       expr::Vmmul<VecDim, Block0, Block1> const>
{
  typedef expr::Vmmul<VecDim, Block0, Block1> const block_type;

  static bool value(MapT const& map, block_type& block)
  {
    // Dispatch_assign only calls Is_par_same_map for distributed
    // expressions.
    assert(!Is_local_only<MapT>::value);

    return 
      // Case 1a: vector is global
      (Is_par_same_map<1, Global_map<1>, Block0>::value(
			Global_map<1>(), block.get_vblk()) &&
       map.num_subblocks(1-VecDim) == 1 &&
       Is_par_same_map<MapDim, MapT, Block1>::value(map, block.get_mblk())) ||

      // Case 1b: vector is replicated
      (Is_par_same_map<1, Replicated_map<1>, Block0>::value(
			Replicated_map<1>(), block.get_vblk()) &&
       map.num_subblocks(1-VecDim) == 1 &&
       Is_par_same_map<MapDim, MapT, Block1>::value(map, block.get_mblk())) ||

      // Case 2:
      (map.num_subblocks(VecDim) == 1 &&
       Is_par_same_map<1, typename Map_project_1<VecDim, MapT>::type, Block0>
	    ::value(Map_project_1<VecDim, MapT>::project(map, 0),
		    block.get_vblk()) &&
       Is_par_same_map<MapDim, MapT, Block1>::value(map, block.get_mblk()));
  }
};



template <dimension_type VecDim,
	  typename       Block0,
	  typename       Block1>
struct Is_par_reorg_ok<expr::Vmmul<VecDim, Block0, Block1> const>
{
  static bool const value = false;
};

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_CORE_EXPR_VMMUL_BLOCK_HPP
