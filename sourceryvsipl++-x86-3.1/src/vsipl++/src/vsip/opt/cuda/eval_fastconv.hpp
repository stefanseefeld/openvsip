/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description
///   CUDA evaluator for fast convolution expressions

#ifndef VSIP_OPT_CUDA_EVAL_FASTCONV_HPP
#define VSIP_OPT_CUDA_EVAL_FASTCONV_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/fft.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/opt/cuda/fastconv.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator for LHS = inv_fftm(vmmul(V, fftm(M)))
///  where V is a vector of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
 	  template <typename> class F2, typename V>
struct Evaluator<op::assign<2>, be::cuda,
  void(LHS &,
       expr::Unary<F2,
         expr::Vmmul<0, V,
         expr::Unary<F1, M> const> const> const &)>
{
  static char const *name() { return "cuda - fast conv, vec coeff";}

  typedef expr::Unary<F1, M> fft_matblock_type;
  typedef expr::Vmmul<0, V, fft_matblock_type const> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  static storage_format_type const storage_format = get_block_layout<LHS>::storage_format;
  typedef impl::cuda::Fastconv<1, complex<float>, storage_format> fconv_type;

  static bool const ct_valid = 
    is_same<complex<float>, typename V::value_type>::value &&
    is_same<complex<float>, typename M::value_type>::value &&
    dda::Data<V, dda::in>::ct_cost == 0 &&
    dda::Data<M, dda::in>::ct_cost == 0 &&
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Vmmul<0, V, expr::Unary<F1, M> const> const> >::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<V, dda::in> ext_kernel(rhs.arg().get_vblk());
    dda::Data<M, dda::in> ext_in(rhs.arg().get_mblk().arg());
    dda::Data<LHS, dda::out> ext_out(lhs);

    return (fconv_type::rt_valid_size(lhs.size(2, 1)) &&
	    ext_kernel.stride(0) == 1 &&
	    ext_in.stride(1) == 1 &&
	    ext_out.stride(1) == 1);
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename V::value_type v_value_type;
    typedef typename M::value_type m_value_type;

    length_type cols = lhs.size(2, 1);
    Matrix<lhs_value_type> tmp(1, cols);

    Vector<v_value_type, V> w(const_cast<V&>(rhs.arg().get_vblk()));
    Matrix<m_value_type, M> in(const_cast<M&>(rhs.arg().get_mblk().arg()));
    Matrix<lhs_value_type, LHS> out(lhs);

    fconv_type fconv(w, cols, false);
    fconv(in, out);
  }
};

/// Evaluator for LHS = inv_fftm(C * fftm(M))
///  where C is a matrix of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
	  template <typename> class F2, typename C>
struct Evaluator<op::assign<2>, be::cuda,
  void(LHS &,
       expr::Unary<F2, expr::Binary<expr::op::Mult, C,
       expr::Unary<F1, M> const, true> const> const &)>
{
  static char const *name() { return "cuda - fast conv, mat coeff(1)";}

  typedef expr::Unary<F1, M> fftm_matblock_type;
  typedef expr::Binary<expr::op::Mult, C, fftm_matblock_type const, true> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;
  static storage_format_type const storage_format = get_block_layout<LHS>::storage_format;
  typedef impl::cuda::Fastconv_base<2, complex<float>, storage_format> fconv_type;

  static bool const ct_valid = 
    is_same<complex<float>, typename C::value_type>::value &&
    is_same<complex<float>, typename M::value_type>::value &&
    dda::Data<C, dda::in>::ct_cost == 0 &&
    dda::Data<M, dda::in>::ct_cost == 0 &&
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Binary<expr::op::Mult, 
      C, expr::Unary<F1, M> const, true> const> >::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<C, dda::in> ext_kernel(rhs.arg().arg1());
    dda::Data<M, dda::in> ext_in(rhs.arg().arg2().arg());
    dda::Data<LHS, dda::out> ext_out(lhs);

    return (fconv_type::rt_valid_size(lhs.size(2, 1)) &&
	    ext_kernel.stride(1) == 1 &&
	    ext_in.stride(1) == 1 &&
	    ext_out.stride(1) == 1);
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename C::value_type c_value_type;
    typedef typename M::value_type m_value_type;

    length_type cols = lhs.size(2, 1);
    Matrix<c_value_type, C> w(const_cast<C&>(rhs.arg().arg1()));
    Matrix<m_value_type, M> in(const_cast<M&>(rhs.arg().arg2().arg()));
    Matrix<lhs_value_type, LHS> out(lhs);

    fconv_type fconv(cols, false);
    fconv.convolve(in, w, out);
  }
};

/// Evaluator for LHS = inv_fftm(fftm(M) * C)
///  where C is a matrix of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
	  template <typename> class F2, typename C>
struct Evaluator<op::assign<2>, be::cuda,
  void(LHS &,
       expr::Unary<F2,
         expr::Binary<expr::op::Mult,
         expr::Unary<F1, M> const, C, true> const> const &)>
{
  static char const *name() { return "cuda - fast conv, mat coeff(2)";}

  typedef expr::Unary<F1, M> fftm_block_type;
  typedef expr::Binary<expr::op::Mult, fftm_block_type const, C, true> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;
  static storage_format_type const storage_format = get_block_layout<LHS>::storage_format;
  typedef impl::cuda::Fastconv_base<2, complex<float>, storage_format> fconv_type;

  static bool const ct_valid = 
    is_same<complex<float>, typename C::value_type>::value &&
    is_same<complex<float>, typename M::value_type>::value &&
    dda::Data<C, dda::in>::ct_cost == 0 &&
    dda::Data<M, dda::in>::ct_cost == 0 &&
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Binary<expr::op::Mult, 
      expr::Unary<F1, M> const, C, true> const> >::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    dda::Data<C, dda::in> ext_kernel(rhs.arg().arg2());
    dda::Data<M, dda::in> ext_in(rhs.arg().arg1().arg());
    dda::Data<LHS, dda::out> ext_out(lhs);

    return (fconv_type::rt_valid_size(lhs.size(2, 1)) &&
	    ext_kernel.stride(1) == 1 &&
	    ext_in.stride(1) == 1 &&
	    ext_out.stride(1) == 1);
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef typename LHS::value_type lhs_value_type;
    typedef typename C::value_type c_value_type;
    typedef typename M::value_type m_value_type;

    length_type cols = lhs.size(2, 1);
    Matrix<c_value_type, C> w(const_cast<C&>(rhs.arg().arg2()));
    Matrix<m_value_type, M> in (const_cast<M&>(rhs.arg().arg1().arg()));
    Matrix<lhs_value_type, LHS> out(lhs);

    fconv_type fconv(cols, false);

    fconv.convolve(in, w, out);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
