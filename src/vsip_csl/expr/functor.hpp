/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/expr/functor.hpp
    @author  Stefan Seefeld
    @date    2009-05-19
    @brief   VSIPL++ Library: custom expression support.
*/

#ifndef VSIP_CSL_EXPR_FUNCTOR_HPP
#define VSIP_CSL_EXPR_FUNCTOR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/view_fwd.hpp>
#include <vsip/core/expr/unary_block.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
using vsip::impl::View_block_storage;

namespace expr
{
using namespace vsip::impl::expr;

/// Convenience base class for unary block functors.
///
/// Template parameters:
///   :ArgumentBlockType: the argument block-type.
template <typename ArgumentBlockType>
class Unary_functor
{
  typedef typename View_block_storage<ArgumentBlockType>::expr_type arg_storage;
public:
  static vsip::dimension_type const dim = ArgumentBlockType::dim;
  typedef typename ArgumentBlockType::value_type value_type;
  typedef value_type result_type;
  typedef typename ArgumentBlockType::map_type map_type;


  Unary_functor(ArgumentBlockType const &a) : arg_(a) {}

  ArgumentBlockType const &arg() const { return arg_;}
  vsip::length_type size() const { return arg_.size();}
  vsip::length_type size(vsip::dimension_type block_dim,
			 vsip::dimension_type d) const
  { return arg_.size(block_dim, d);}
  map_type const &map() const { return arg_.map();}

  template <typename ResultBlockType>
  void apply(ResultBlockType &r) const {}

private:
  arg_storage arg_;
};

/// Convenience base class for binary block functors.
///
/// Template parameters:
///   :Argument1BlockType: the first argument block-type.
///   :Argument2BlockType: the second argument block-type.
template <typename Argument1BlockType,
	  typename Argument2BlockType>
class Binary_functor
{
  typedef typename View_block_storage<Argument1BlockType>::expr_type 
  arg1_storage;
  typedef typename View_block_storage<Argument2BlockType>::expr_type 
  arg2_storage;
public:
  static vsip::dimension_type const dim = Argument1BlockType::dim;
  typedef typename Argument1BlockType::map_type map_type;
    
  Binary_functor(Argument1BlockType const &a1, Argument2BlockType const &a2)
    : arg1_(a1), arg2_(a2) 
  {
  }

  Argument1BlockType const &arg1() const { return arg1_;}
  Argument2BlockType const &arg2() const { return arg2_;}
  vsip::length_type size() const { return arg1.size();}
  vsip::length_type size(vsip::dimension_type block_dim,
			 vsip::dimension_type d) const
  { return arg1.size(block_dim, d);}
  map_type const &map() const { return arg1.map();}

  template <typename ResultBlockType>
  void apply(ResultBlockType &r) const {}

private:
  arg1_storage arg1_;
  arg2_storage arg2_;
};


} // namespace vsip_csl::expr
} // namespace vsip_csl

#endif
