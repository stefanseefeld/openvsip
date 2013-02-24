/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/eval_freqswap.hpp
    @author  Don McCoy
    @date    2009-07-02
    @brief   VSIPL++ Library: CUDA-based freqswap evaluators (aka fftshift)
*/

#ifndef VSIP_OPT_CUDA_FREQSWAP_HPP
#define VSIP_OPT_CUDA_FREQSWAP_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/dda.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/dda.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// CUDA evaluator for freqswap function 
template <typename Block1,
          typename Block2>
struct Evaluator<op::freqswap, be::cuda, void(Block1 const&, Block2&)>
{
  typedef typename get_block_layout<Block1>::order_type in_order_type;
  typedef typename get_block_layout<Block2>::order_type out_order_type;

  static bool const ct_valid =
    // views must be matrices
    Block1::dim == 2 &&
    Block2::dim == 2 &&
    // types must be supported and equal
    impl::cuda::Traits<typename Block1::value_type>::valid &&
    is_same<typename Block1::value_type, typename Block2::value_type>::value &&
    // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0 &&
    // check that format is interleaved.
    !impl::is_split_block<Block1>::value &&
    !impl::is_split_block<Block2>::value;

  static bool rt_valid(Block1 const& in, Block2& out) 
  { 
    dda::Data<Block1, dda::in> data_in(in);
    dda::Data<Block2, dda::out> data_out(out);

    // check that data is unit stride and dense
    dimension_type const in_dim0 = in_order_type::impl_dim0;
    dimension_type const in_dim1 = in_order_type::impl_dim1;
    dimension_type const out_dim0 = out_order_type::impl_dim0;
    dimension_type const out_dim1 = out_order_type::impl_dim1;

    return 
      (data_in.stride(in_dim1) == 1) &&
      (data_out.stride(out_dim1) == 1) &&
      (data_in.stride(in_dim0) == static_cast<stride_type>(data_in.size(in_dim1))) &&
      (data_out.stride(out_dim0) == static_cast<stride_type>(data_out.size(out_dim1)));
  }

  static void exec(Block1 const& in, Block2& out)
  {
    assert(in.size(2, 0) == out.size(2, 0));
    assert(in.size(2, 1) == out.size(2, 1));

    impl::cuda::dda::Data<Block1, dda::in> dev_in(in);
    impl::cuda::dda::Data<Block2, dda::out> dev_out(out);

    impl::cuda::fftshift(dev_in.ptr(), 
			 dev_out.ptr(), 
			 dev_out.size(0), 
			 dev_out.size(1),
			 in_order_type::impl_dim0,
			 out_order_type::impl_dim0);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CUDA_FREQSWAP_HPP
