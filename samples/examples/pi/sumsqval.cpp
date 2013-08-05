/* Copyright (c) 2010, 2011 CodeSourcery, Inc.  All rights reserved. */

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
///   sumsqval using parallel iterator expression

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/pi.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/profile.hpp>

using namespace vsip;
using namespace vsip_csl;

namespace vsip_csl
{
namespace dispatcher
{
/// Provide a custom evaluator that targets specifically
/// output(i) = sumsqval(input.row(i)))
template <typename B, typename I>
struct Evaluator<op::pi_assign, be::user,
  void(pi::Call<B, I> &,
       pi::Unary<pi::Sumsqval, 
		 pi::Call<Dense<2u, complex<float>, tuple<0u, 1u, 2u> >,
			  I, whole_domain_type> > const &)>
{
  typedef pi::Call<B, I> LHS;
  typedef pi::Unary<pi::Sumsqval, 
		    pi::Call<Dense<2u, complex<float>, tuple<0u, 1u, 2u> >,
			     I, whole_domain_type> > RHS;
  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef Dense<2, complex<float> > rhs_block_type;

    B &lhs_block = lhs.block();
    // Extract the block from the expression, ...
    rhs_block_type const &rhs_block = rhs.arg().block();
    // ...wrap it in a matrix, ...
    Matrix<complex<float> > matrix(const_cast<rhs_block_type &>(rhs_block));
    // ...and explicitly iterate over its rows.
    for (index_type i = 0; i < lhs_block.size(); ++i)
      lhs_block.put(i, sumsqval(matrix.row(i)));
  }

};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl


int main(int argc, char **argv)
{
  vsipl init(argc, argv);

  int const N = 8;

  Matrix<complex<float> > input(N, N);
  for (index_type r = 0; r != input.size(0); ++r)
    input.row(r) = ramp(static_cast<float>(N*r), 1.f, N);

  Vector<complex<float> > output(N);
  Vector<complex<float> > ref(N);

  profile::Timer t;
  t.start();
  for (int i=0; i < ref.size(); i++)
    ref.put(i, sumsqval(input.row(i)));
  t.stop();
  std::cout << "serial version : " << t.delta() << " seconds" << std::endl;

  pi::Iterator<> i;
  t.start();
  output(i) = sumsqval(input.row(i));
  t.stop();
  std::cout << "parallel iterator version : " << t.delta() << " seconds" << std::endl;

  std::cout << "input : \n" << input << std::endl;
  std::cout << "output : \n" << output << std::endl;
  std::cout << "ref : \n" << ref << std::endl;
}
