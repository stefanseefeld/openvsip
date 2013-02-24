/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/cbe/ppu/eval_fastconv.hpp
    @author  Jules Bergmann
    @date    2007-03-05
    @brief   VSIPL++ Library: General evaluator for fast convolution

*/

#ifndef VSIP_OPT_CBE_PPU_EVAL_FASTCONV_HPP
#define VSIP_OPT_CBE_PPU_EVAL_FASTCONV_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cbe/ppu/fastconv.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator for LHS = inv_fftm(vmmul(V, fftm(M)))
///  where V is a vector of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
 	  template <typename> class F2, typename V>
struct Evaluator<op::assign<2>, be::cbe_sdk,
  void(LHS &,
       expr::Unary<F2,
         expr::Vmmul<0, V,
	   expr::Unary<F1, M> const> const> const &)>
{
  static char const* name() { return "Cbe_sdk_tag"; }

  typedef expr::Unary<F1, M> fft_matblock_type;
  typedef expr::Vmmul<0, V, fft_matblock_type const> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  static storage_format_type const storage_format = get_block_layout<LHS>::storage_format;
  typedef impl::cbe::Fastconv<1, complex<float>, storage_format> fconv_type;

  static bool const ct_valid = 
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<
      expr::Vmmul<0, V, expr::Unary<F1, M> const> const> >::value &&
    is_same<complex<float>, typename V::value_type>::value &&
    is_same<complex<float>, typename M::value_type>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<V, dda::in> data_kernel(rhs.arg().get_vblk());
    dda::Data<M, dda::in> data_in(rhs.arg().get_mblk().arg());
    dda::Data<LHS, dda::out> data_out(lhs);

    return (fconv_type::is_size_valid(lhs.size(2, 1)) &&
	    data_kernel.stride(0) == 1 &&
	    data_in.stride(1) == 1 &&
	    data_out.stride(1) == 1);
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    Vector<complex<float>, V> w 
      (const_cast<V&>(rhs.arg().get_vblk()));
    Matrix<complex<float>, M> in 
      (const_cast<M&>(rhs.arg().get_mblk().arg()));
    Matrix<complex<float>, LHS> out(lhs);

    fconv_type fconv(w, out.size(1), false);
    fconv(in, out);
  }
};



/// Evaluator for LHS = inv_fftm(C * fftm(M))
///  where C is a matrix of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
	  template <typename> class F2, typename C>
struct Evaluator<op::assign<2>, be::cbe_sdk,
  void(LHS &,
       expr::Unary<F2, expr::Binary<expr::op::Mult, C,
	 expr::Unary<F1, M> const, true> const> const &)>
{
  static char const* name() { return "Cbe_sdk_tag"; }

  typedef expr::Unary<F1, M> fftm_matblock_type;
  typedef expr::Binary<expr::op::Mult, C, fftm_matblock_type const, true> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  static storage_format_type const storage_format = get_block_layout<LHS>::storage_format;
  typedef impl::cbe::Fastconv<2, complex<float>, storage_format> fconv_type;

  static bool const ct_valid = 
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Binary<expr::op::Mult, 
      C, expr::Unary<F1, M> const, true> const> >::value &&
    is_same<complex<float>, typename C::value_type>::value &&
    is_same<complex<float>, typename M::value_type>::value &&
    dda::Data<C, dda::in>::ct_cost == 0 &&
    dda::Data<M, dda::in>::ct_cost == 0 &&
    dda::Data<LHS, dda::out>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<C, dda::in> data_kernel(rhs.arg().arg1());
    dda::Data<M, dda::in> data_in(rhs.arg().arg2().arg());
    dda::Data<LHS, dda::out> data_out(lhs);

    return (fconv_type::is_size_valid(lhs.size(2, 1)) &&
	    data_kernel.stride(1) == 1 &&
	    data_in.stride(1) == 1 &&
	    data_out.stride(1) == 1);
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    Matrix<complex<float>, C> w 
      (const_cast<C&>(rhs.arg().arg1()));
    Matrix<complex<float>, M> in 
      (const_cast<M&>(rhs.arg().arg2().arg()));
    Matrix<complex<float>, LHS> out(lhs);

    fconv_type fconv(w, in.size(1), false);
    fconv(in, out);
  }
};


/// Evaluator for LHS = inv_fftm(fftm(M) * C)
///  where C is a matrix of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
	  template <typename> class F2, typename C>
struct Evaluator<op::assign<2>, be::cbe_sdk,
  void(LHS &,
       expr::Unary<F2,
         expr::Binary<expr::op::Mult,
	   expr::Unary<F1, M> const, C, true> const> const &)>
{
  static char const* name() { return "Cbe_sdk_tag";}

  typedef expr::Unary<F1, M> fftm_block_type;
  typedef expr::Binary<expr::op::Mult, fftm_block_type const, C, true> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  static storage_format_type const storage_format = get_block_layout<LHS>::storage_format;
  typedef impl::cbe::Fastconv<2, complex<float>, storage_format> fconv_type;

  static bool const ct_valid = 
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Binary<expr::op::Mult, 
      expr::Unary<F1, M> const, C, true> const> >::value &&
    is_same<complex<float>, typename C::value_type>::value &&
    is_same<complex<float>, typename M::value_type>::value &&
    dda::Data<C, dda::in>::ct_cost == 0 &&
    dda::Data<M, dda::in>::ct_cost == 0 &&
    dda::Data<LHS, dda::out>::ct_cost == 0;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<C, dda::in> data_kernel(rhs.arg().arg2());
    dda::Data<M, dda::in> data_in(rhs.arg().arg1().arg());
    dda::Data<LHS, dda::out> data_out(lhs);

    return (fconv_type::is_size_valid(lhs.size(2, 1)) &&
	    data_kernel.stride(1) == 1 &&
	    data_in.stride(1) == 1 &&
	    data_out.stride(1) == 1);
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type cols = lhs.size(2, 1);
    Matrix<complex<float>, C> w 
      (const_cast<C&>(rhs.arg().arg2()));
    Matrix<complex<float>, M> in 
      (const_cast<M&>(rhs.arg().arg1().arg()));
    Matrix<complex<float>, LHS> out(lhs);

    fconv_type fconv(w, cols, false);
    fconv(in, out);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
