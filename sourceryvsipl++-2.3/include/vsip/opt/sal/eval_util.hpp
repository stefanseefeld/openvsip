/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/eval_util.hpp
    @author  Jules Bergmann
    @date    2006-05-26
    @brief   VSIPL++ Library: Util routines for Mercury SAL Dispatch.
*/

#ifndef VSIP_OPT_SAL_EVAL_UTIL_HPP
#define VSIP_OPT_SAL_EVAL_UTIL_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/extdata.hpp>
#include <vsip/core/coverage.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace sal
{

/// Helper class for Effective_value_type.

template <typename T,
	  typename ComplexFmt,
	  bool     IsVector = true>
struct Effective_type
{
  typedef typename ITE_Type<IsVector, As_type<T*>, As_type<T> >::type type;
};

template <typename T,
	  bool     IsVector>
struct Effective_type<complex<T>, Cmplx_split_fmt, IsVector>
{
  typedef typename ITE_Type<IsVector, As_type<std::pair<T*, T*> >,
			              As_type<std::pair<T,  T > > >::type type;
};



/// Determine the effective value type of a block.
///
/// For scalar_blocks, the effective value type is the block's value
/// type.
///
/// For split complex blocks, the effective type is a std::pair
/// of pointers.
///
/// For other blocks, the effective type is a pointer to the block's
/// value_type.

template <typename BlockT,
	  typename ValueT = typename BlockT::value_type>
struct Effective_value_type
{
  typedef typename Block_layout<BlockT>::complex_type complex_type;
  typedef typename Effective_type<ValueT, complex_type, true>::type type;
};

template <dimension_type Dim,
	  typename       T>
struct Effective_value_type<expr::Scalar<Dim, T>, T>
{
  typedef typename Effective_type<T, Cmplx_inter_fmt, false>::type type;
};



template <typename BlockT,
	  typename LP = typename Block_layout<BlockT>::layout_type>
struct Ext_wrapper
{
  typedef typename Block_layout<BlockT>::complex_type complex_type;
  typedef typename BlockT::value_type                 value_type;
  typedef typename ITE_Type<Is_complex<value_type>::value,
                            As_type<complex_type>,
                            As_type<Any_type> >::type use_complex_type;
  typedef Sal_vector<value_type, use_complex_type>    sal_type;

  Ext_wrapper(BlockT& block, sync_action_type sync)
    : ext_(block, sync)
  {}

  Ext_wrapper(BlockT const& block, sync_action_type sync)
    : ext_(block, sync)
  {}

  bool is_unit_stride() { return ext_.stride(0) == 1; }

  Ext_data<BlockT, LP> ext_;
};



template <dimension_type Dim,
	  typename       T,
	  typename       LP>
struct Ext_wrapper<expr::Scalar<Dim, T>, LP>
{
  typedef expr::Scalar<Dim, T> block_type;
  typedef Sal_scalar<T>        sal_type;

  Ext_wrapper(block_type& block, sync_action_type /*sync*/)
    : value_(block.value())
  {}

  Ext_wrapper(block_type const& block, sync_action_type /*sync*/)
    : value_(block.value())
  {}

  bool is_unit_stride() { return true; }

  T value_;
};



template <dimension_type Dim,
	  typename       T,
	  typename       LP>
struct Ext_wrapper<expr::Scalar<Dim, T> const, LP>
{
  typedef expr::Scalar<Dim, T> block_type;
  typedef Sal_scalar<T>        sal_type;

  Ext_wrapper(block_type& block, sync_action_type /*sync*/)
    : value_(block.value())
  {}

  Ext_wrapper(block_type const& block, sync_action_type /*sync*/)
    : value_(block.value())
  {}

  bool is_unit_stride() { return true; }

  T value_;
};



} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SAL_EVAL_UTIL_HPP
