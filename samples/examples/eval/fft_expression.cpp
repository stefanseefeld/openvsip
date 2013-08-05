/* Copyright (c) 2009, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

/// Description
///   Custom expression evaluator for a FFT expression.
///   Demonstrate how to disabiguate multiple FFT functors.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <iostream>

namespace vsip_csl
{
namespace dispatcher
{
// Evaluator for LHS = C * fftm(M)
// From the fact that this is a 2D assignment we know that fft(M) generates
// a Matrix. The ''is_fftm_functor'' filter helps to disambiguate between
// 2D FFT and FFTM, which both operate on and return matrices.
template <typename LHS, typename C,
	  template <typename> class F, typename M>
struct Evaluator<op::assign<2>, be::user,
  void(LHS &, expr::Binary<expr::op::Mult, C, expr::Unary<F, M> const, true> const &),
  typename enable_if<expr::is_fftm_functor<F<M> > >::type>
{
  static char const* name() { return "be::user C*fftm(M)";}

  typedef expr::Unary<F, M> fftm_block_type;
  typedef expr::Binary<expr::op::Mult, C, fftm_block_type const, true> RHS;

  typedef F<M> fftm_type;
  typedef typename fftm_type::backend_type fwd_backend_type;
  typedef typename fftm_type::workspace_type fwd_workspace_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static bool const ct_valid = true;

  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs ATTRIBUTE_UNUSED, RHS const &rhs ATTRIBUTE_UNUSED)
  {
    std::cout << "Executing evaluator " << name() << std::endl;
  }
};

// Evaluator for LHS = C * fft(M)
// From the fact that this is a 2D assignment we know that fft(M) generates
// a Matrix. The ''is_fft_functor'' filter helps to disambiguate between
// 2D FFT and FFTM, which both operate on and return matrices.
template <typename LHS, typename C,
	  template <typename> class F, typename M>
struct Evaluator<op::assign<2>, be::user,
  void(LHS &, expr::Binary<expr::op::Mult, C, expr::Unary<F, M> const, true> const &),
  typename enable_if<expr::is_fft_functor<F<M> > >::type>
{
  static char const* name() { return "be::user C*fft(M)";}

  typedef expr::Unary<F, M> fftm_block_type;
  typedef expr::Binary<expr::op::Mult, C, fftm_block_type const, true> RHS;

  typedef F<M> fftm_type;
  typedef typename fftm_type::backend_type fwd_backend_type;
  typedef typename fftm_type::workspace_type fwd_workspace_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static bool const ct_valid = true;

  static bool rt_valid(LHS &, RHS const &) { return true;}
  
  static void exec(LHS &lhs ATTRIBUTE_UNUSED, RHS const &rhs ATTRIBUTE_UNUSED)
  {
    std::cout << "Executing evaluator " << name() << std::endl;
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl



int 
main(int argc, char **argv)
{
  using namespace vsip;

  typedef complex<float> C;
  typedef Fft<const_Matrix, C, C, fft_fwd> fft_type;
  typedef Fftm<C, C, row, fft_fwd> fftm_type;

  vsipl init(argc, argv);

  length_type M = 64;
  length_type N = 1024;

  // Create the FFT object.
  fft_type fft(Domain<2>(M, N), 1.);

  // Create the FFTM object.
  fftm_type fftm(Domain<2>(M, N), 1.0);

  // Create the data.
  Matrix<C> in(M, N, 1.);
  Matrix<C> c(M, N, 1.);
  Matrix<C> out(M, N);

  std::cout << "invoking C * fft(in)" << std::endl; 
  out = c * fft(in);
  std::cout << "invoking C * fftm(in)" << std::endl; 
  out = c * fftm(in);
}
