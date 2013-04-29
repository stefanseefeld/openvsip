/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/expr/eval_fastconv.hpp
    @author  Jules Bergmann
    @date    2007-02-02
    @brief   VSIPL++ Library: General evaluator for fast convolution

*/

#ifndef VSIP_OPT_EXPR_EVAL_FASTCONV_HPP
#define VSIP_OPT_EXPR_EVAL_FASTCONV_HPP

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/fft.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator for LHS = inv_fftm(vmmul(V, fftm(M)))
///  where V is a vector of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
 	  template <typename> class F2, typename V>
struct Evaluator<op::assign<2>, be::fc_expr,
  void(LHS &,
       expr::Unary<F2,
         expr::Vmmul<0, V,
           expr::Unary<F1, M> const> const> const &),
  typename enable_if_c<
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<
      expr::Vmmul<0, V, expr::Unary<F1, M> const> const> >::value>::type>
{
  static char const *name() { return "fc_expr-vw";}

  typedef expr::Unary<F1, M> fft_matblock_type;
  typedef expr::Vmmul<0, V, fft_matblock_type const> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  typedef F1<M> fwd_fft_type;
  typedef typename fwd_fft_type::backend_type fwd_backend_type;
  typedef typename fwd_fft_type::workspace_type fwd_workspace_type;

  typedef F2<inv_block_type const> inv_fft_type;
  typedef typename inv_fft_type::backend_type inv_backend_type;
  typedef typename inv_fft_type::workspace_type inv_workspace_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename fft_matblock_type::value_type tmp_value_type;

  static bool const ct_valid = true;

  static bool rt_valid(LHS &, RHS const &rhs)
  {
#if 0
    // Check if the evaluator supports the scaling implied by
    // the FFTs.
    //
    // This evaluator uses the FFTs directly, so it will implicitly
    // follow the requested scaling.  However, if this evaluator is
    // adapted to use other fast convolution implementations that
    // have limited scaling (such as only unit), this check will
    // be necessary.
    
    typedef typename impl::scalar_of<T>::type scalar_type;

    fwd_workspace_type const &fwd_workspace(rhs.arg().get_mblk().operation().workspace());
    inv_workspace_type const &inv_workspace(rhs.operation().workspace());
    // Check FFT scaling totals 1
    return almost_equal(fwd_workspace.scale() * inv_workspace.scale(), scalar_type(1));
#else
    (void)rhs;
    return true;
#endif
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type rows = rhs.arg().get_mblk().size(2, 0);
    length_type cols = rhs.arg().get_mblk().size(2, 1);
    Matrix<tmp_value_type> tmp(rows, cols);

    Vector<typename V::value_type, V> w(const_cast<V &>(rhs.arg().get_vblk()));
    Matrix<typename M::value_type, M> in(const_cast<M &>(rhs.arg().get_mblk().arg()));
    Matrix<lhs_value_type, LHS> out(lhs);

    fwd_workspace_type const &fwd_workspace (rhs.arg().get_mblk().operation().workspace());
    fwd_backend_type &fwd_backend (const_cast<fwd_backend_type&>(rhs.arg().get_mblk().operation().backend()));

    inv_workspace_type const &inv_workspace(rhs.operation().workspace());
    inv_backend_type &inv_backend(const_cast<inv_backend_type&>(rhs.operation().backend()));

    fwd_workspace.out_of_place(&fwd_backend, in, tmp);
    for (index_type r=0; r<rows; ++r)
      tmp.row(r) *= w;
    inv_workspace.out_of_place(&inv_backend, tmp, out);
  }
};

/// Evaluator for LHS = inv_fftm(C * fftm(M))
///  where C is a matrix of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
	  template <typename> class F2, typename C>
struct Evaluator<op::assign<2>, be::fc_expr,
  void(LHS &,
       expr::Unary<F2, expr::Binary<expr::op::Mult, C,
       expr::Unary<F1, M> const, true> const> const &),
  typename enable_if_c<
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Binary<expr::op::Mult, 
      C, expr::Unary<F1, M> const, true> const> >::value>::type>
{
  static char const* name() { return "Fc_expr_tag-mwl"; }

  typedef expr::Unary<F1, M> fftm_matblock_type;
  typedef expr::Binary<expr::op::Mult, C, fftm_matblock_type const, true> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  typedef F1<M> fwd_fftm_type;
  typedef typename fwd_fftm_type::backend_type fwd_backend_type;
  typedef typename fwd_fftm_type::workspace_type fwd_workspace_type;

  typedef F2<inv_block_type const> inv_fftm_type;
  typedef typename inv_fftm_type::backend_type inv_backend_type;
  typedef typename inv_fftm_type::workspace_type inv_workspace_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename fftm_matblock_type::value_type tmp_value_type;

  static bool const ct_valid = true;

  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type rows = rhs.arg().arg1().size(2, 0);
    length_type cols = rhs.arg().arg1().size(2, 1);
    Matrix<tmp_value_type> tmp(rows, cols);

    Matrix<typename C::value_type, C>
      w(const_cast<C &>(rhs.arg().arg1()));
    Matrix<typename M::value_type, M>
      in(const_cast<M &>(rhs.arg().arg2().arg()));
    Matrix<lhs_value_type, LHS> out(lhs);

    fwd_workspace_type const &fwd_workspace (rhs.arg().arg2().operation().workspace());
    fwd_backend_type &fwd_backend (const_cast<fwd_backend_type&>(rhs.arg().arg2().operation().backend()));

    inv_workspace_type const &inv_workspace(rhs.operation().workspace());
    inv_backend_type &inv_backend (const_cast<inv_backend_type&>(rhs.operation().backend()));

    fwd_workspace.out_of_place(&fwd_backend, in, tmp);
    tmp *= w;
    inv_workspace.out_of_place(&inv_backend, tmp, out);
  }
};

/// Evaluator for LHS = inv_fftm(fftm(M) * C)
///  where C is a matrix of coefficients
template <typename LHS,
	  template <typename> class F1, typename M,
	  template <typename> class F2, typename C>
struct Evaluator<op::assign<2>, be::fc_expr,
  void(LHS &,
       expr::Unary<F2,
         expr::Binary<expr::op::Mult,
           expr::Unary<F1, M> const, C, true> const> const &),
  typename enable_if_c<
    expr::is_fftm_functor<F1<M> >::value &&
    expr::is_fftm_functor<F2<expr::Binary<expr::op::Mult, 
      expr::Unary<F1, M> const, C, true> const> >::value>::type>
{
  static char const *name() { return "fc_expr-mwr";}

  typedef expr::Unary<F1, M> fftm_block_type;
  typedef expr::Binary<expr::op::Mult, fftm_block_type const, C, true> inv_block_type;
  typedef expr::Unary<F2, inv_block_type const> RHS;

  typedef F1<M> fwd_fftm_type;
  typedef typename fwd_fftm_type::backend_type fwd_backend_type;
  typedef typename fwd_fftm_type::workspace_type fwd_workspace_type;

  typedef F2<inv_block_type const> inv_fftm_type;
  typedef typename inv_fftm_type::backend_type inv_backend_type;
  typedef typename inv_fftm_type::workspace_type inv_workspace_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename fftm_block_type::value_type tmp_value_type;

  static bool const ct_valid = true;

  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    length_type rows = rhs.arg().arg2().size(2, 0);
    length_type cols = rhs.arg().arg2().size(2, 1);
    Matrix<tmp_value_type> tmp(rows, cols);

    Matrix<typename C::value_type, C> w(const_cast<C &>(rhs.arg().arg2()));
    Matrix<typename M::value_type, M> in(const_cast<M &>(rhs.arg().arg1().arg()));
    Matrix<lhs_value_type, LHS> out(lhs);

    fwd_workspace_type const &fwd_workspace(rhs.arg().arg1().operation().workspace());
    fwd_backend_type &fwd_backend(const_cast<fwd_backend_type &>(rhs.arg().arg1().operation().backend()));
    inv_workspace_type const &inv_workspace(rhs.operation().workspace());
    inv_backend_type &inv_backend(const_cast<inv_backend_type &>(rhs.operation().backend()));

    fwd_workspace.out_of_place(&fwd_backend, in, tmp);
    tmp *= w;
    inv_workspace.out_of_place(&inv_backend, tmp, out);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
