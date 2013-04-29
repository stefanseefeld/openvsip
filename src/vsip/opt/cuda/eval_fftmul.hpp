/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cuda/eval_fftmul.hpp
    @author  Don McCoy
    @date    2009-07-20
    @brief   VSIPL++ Library: Evaluator for an FFTM followed by
               multiplication with another matrix
*/

#ifndef VSIP_OPT_CUDA_EVAL_FFTMUL_HPP
#define VSIP_OPT_CUDA_EVAL_FFTMUL_HPP

#include <vsip/core/fft.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/dda.hpp>

namespace vsip_csl
{
namespace dispatcher
{

template <typename LHS,
          typename LBlock,
          typename RBlock,
          template <typename> class F>
struct Evaluator<op::assign<2>, be::cuda,
  void(LHS &,
       expr::Binary<expr::op::Mult, 
         LBlock,
         expr::Unary<F,
           RBlock,
           false> const,
         true> const &)>
{
  static char const* name() { return "Cuda_tag"; }

  typedef
  expr::Binary<expr::op::Mult, 
    LBlock,
    expr::Unary<F, 
      RBlock,
      false> const,
    true> const
  RHS;

  typedef std::complex<float> C;
  typedef typename impl::fft::workspace<2u, C, C>  workspace_type;
  typedef typename impl::fft::Fftm_backend<C, C, vsip::row, fft_fwd>  be_fwd_row_type;
  typedef typename impl::fft::Fftm_backend<C, C, vsip::col, fft_fwd>  be_fwd_col_type;
  typedef typename impl::fft::Fftm_backend<C, C, vsip::row, fft_inv>  be_inv_row_type;
  typedef typename impl::fft::Fftm_backend<C, C, vsip::col, fft_inv>  be_inv_col_type;
  typedef typename expr::op::fft<2u, 
    be_fwd_row_type, workspace_type>::Functor<RBlock> fftm_fwd_row_functor_type;
  typedef typename expr::op::fft<2u, 
    be_fwd_col_type, workspace_type>::Functor<RBlock> fftm_fwd_col_functor_type;
  typedef typename expr::op::fft<2u, 
    be_inv_row_type, workspace_type>::Functor<RBlock> fftm_inv_row_functor_type;
  typedef typename expr::op::fft<2u, 
    be_inv_col_type, workspace_type>::Functor<RBlock> fftm_inv_col_functor_type;

  typedef F<RBlock> unary_functor_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  typedef typename get_block_layout<LHS>::order_type lhs_order_type;
  typedef typename get_block_layout<LBlock>::order_type l_order_type;
  typedef typename get_block_layout<RBlock>::order_type r_order_type;

  static bool const ct_valid =
    // the unary functor must be fftm
    (is_same<fftm_fwd_row_functor_type, unary_functor_type>::value ||
     is_same<fftm_fwd_col_functor_type, unary_functor_type>::value ||
     is_same<fftm_inv_row_functor_type, unary_functor_type>::value ||
     is_same<fftm_inv_col_functor_type, unary_functor_type>::value) &&
    // only complex is presently handled
    is_same<lhs_value_type, std::complex<float> >::value &&
    is_same<rhs_value_type, std::complex<float> >::value &&
    // source types must be the same as the result type
    is_same<lhs_value_type, typename LBlock::value_type>::value &&
    is_same<lhs_value_type, typename RBlock::value_type>::value &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<LBlock, dda::in>::ct_cost == 0 &&
    dda::Data<RBlock, dda::in>::ct_cost == 0 &&
    // complex split is not supported presently
    !impl::is_split_block<LHS>::value &&
    !impl::is_split_block<LBlock>::value &&
    !impl::is_split_block<RBlock>::value &&
    // dimension order must be the same
    is_same<lhs_order_type, l_order_type>::value &&
    is_same<lhs_order_type, r_order_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out> data_lhs(lhs);
    dda::Data<LBlock, dda::in> data_l(rhs.arg1());
    dda::Data<RBlock, dda::in> data_r(rhs.arg2().arg());

    dimension_type const dim0 = lhs_order_type::impl_dim0;           
    dimension_type const dim1 = lhs_order_type::impl_dim1;           

    return
      (data_lhs.stride(dim1) == 1) &&
      (data_l.stride(dim1) == 1) &&
      (data_r.stride(dim1) == 1) &&
      (data_lhs.stride(dim0) == static_cast<stride_type>(data_lhs.size(dim1))) &&
      (data_l.stride(dim0) == static_cast<stride_type>(data_l.size(dim1))) &&
      (data_r.stride(dim0) == static_cast<stride_type>(data_r.size(dim1)));
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    // compute the Fftm, store the temporary result in the output block
    rhs.arg2().apply(lhs);

    // complete by multiplying by the other input and returning
    impl::cuda::dda::Data<LHS, dda::inout> dev_lhs(lhs);
    impl::cuda::dda::Data<LBlock, dda::in> dev_l(rhs.arg1());

    dimension_type const dim0 = lhs_order_type::impl_dim0;
    dimension_type const dim1 = lhs_order_type::impl_dim1;

    // matrix-matrix multiply with scaling
    impl::cuda::mmmuls(dev_l.ptr(),    // input
		       dev_lhs.ptr(),  // input
		       dev_lhs.ptr(),  // output
		       1.0f,           // scale factor
		       dev_lhs.size(dim0), dev_lhs.size(dim1));
  }
};

} // namespace vsip::dispatcher
} // namespace vsip

#endif // VSIP_OPT_CUDA_EVAL_FFTMUL_HPP
