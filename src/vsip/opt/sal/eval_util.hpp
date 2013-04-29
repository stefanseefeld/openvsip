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

#include <vsip/dda.hpp>
#include <vsip/core/metaprogramming.hpp>


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

template <typename T, storage_format_type C, bool IsVector = true>
struct Effective_type
{
  typedef typename conditional<IsVector, T*, T>::type type;
};

template <typename T, bool IsVector>
struct Effective_type<complex<T>, split_complex, IsVector>
{
  typedef typename conditional<IsVector, std::pair<T*,T*>, std::pair<T,T> >::type type;
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

template <typename B, typename T = typename B::value_type>
struct Effective_value_type
{
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;
  typedef typename Effective_type<T, storage_format, true>::type type;
};

template <dimension_type D, typename T>
struct Effective_value_type<expr::Scalar<D, T>, T>
{
  typedef typename Effective_type<T, interleaved_complex, false>::type type;
};

template <dimension_type D, typename T>
struct Effective_value_type<expr::Scalar<D, T> const, T>
{
  typedef typename Effective_type<T, interleaved_complex, false>::type type;
};

template <typename B,
	  dda::sync_policy S,
	  typename L = typename get_block_layout<B>::type>
struct DDA_wrapper
{
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;
  static storage_format_type const use_storage_format =
    is_complex<typename B::value_type>::value ? storage_format : any_storage_format;
  typedef Sal_vector<typename B::value_type, use_storage_format> sal_type;

  DDA_wrapper(B& block) : data(block) {}
  bool is_unit_stride() { return data.stride(0) == 1;}
  dda::Data<B, S, L> data;
};

template <typename B, typename L>
struct DDA_wrapper<B, dda::in, L>
{
  static storage_format_type const storage_format = get_block_layout<B>::storage_format;
  static storage_format_type const use_storage_format =
    is_complex<typename B::value_type>::value ? storage_format : any_storage_format;
  typedef const_Sal_vector<typename B::value_type, use_storage_format> sal_type;

  DDA_wrapper(B const &block) : data(block) {}
  bool is_unit_stride() { return data.stride(0) == 1;}
  dda::Data<B, dda::in, L> data;
};

template <dimension_type D, typename T, dda::sync_policy S, typename L>
struct DDA_wrapper<expr::Scalar<D, T>, S, L>
{
  typedef expr::Scalar<D, T> block_type;
  typedef Sal_scalar<T> sal_type;

  DDA_wrapper(block_type &block) : value(block.value()) {}
  bool is_unit_stride() { return true;}
  T value;
};

template <dimension_type D, typename T, typename L>
struct DDA_wrapper<expr::Scalar<D, T>, dda::in, L>
{
  typedef expr::Scalar<D, T> block_type;
  typedef Sal_scalar<T> sal_type;

  DDA_wrapper(block_type const &block) : value(block.value()) {}
  bool is_unit_stride() { return true;}
  T value;
};

template <dimension_type D, typename T, dda::sync_policy S, typename L>
struct DDA_wrapper<expr::Scalar<D, T> const, S, L>
{
  typedef expr::Scalar<D, T> block_type;
  typedef Sal_scalar<T> sal_type;

  DDA_wrapper(block_type &block) : value(block.value()) {}
  bool is_unit_stride() { return true;}
  T value;
};

template <dimension_type D, typename T, typename L>
struct DDA_wrapper<expr::Scalar<D, T> const, dda::in, L>
{
  typedef expr::Scalar<D, T> block_type;
  typedef Sal_scalar<T> sal_type;

  DDA_wrapper(block_type const &block) : value(block.value()) {}
  bool is_unit_stride() { return true;}
  T value;
};

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SAL_EVAL_UTIL_HPP
