/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/expr_ops_info.cpp
    @author  Jules Bergmann
    @date    2006-08-04
    @brief   VSIPL++ Library: Test expression template reductions
             used for generating operation counts and expression tags.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/opt/expr/ops_info.hpp>

#include <vsip_csl/test.hpp>
#include <iostream>

// Test that the ops/point of an EXPR is as expected by OPS.
template <typename ViewT>
void
test_expr_ops(unsigned ops, ViewT /*expr*/)
{
  typedef typename ViewT::block_type block_type;

  test_assert(ops == vsip_csl::expr::Ops_per_point<block_type>::value);
}


// Test that the operation tag generated for the given expression
// is correct.
template <typename ViewT>
void
test_expr_tag(char const* tag, ViewT /*expr*/)
{
  typedef typename ViewT::block_type block_type;
  std::cout << tag << ' ' << vsip::impl::Reduce_expr_op_name::template transform<block_type>::tag() << std::endl;
  test_assert(tag == vsip::impl::Reduce_expr_op_name::template transform<block_type>::tag());
}



void
test_op_counts()
{
  vsip::Vector<float> vec1(5);
  vsip::Vector<float> vec2(5);
  vsip::Vector<std::complex<float> > vec3(5);
  vsip::Vector<std::complex<float> > vec4(5);

  test_expr_ops(1, vec1 + vec2);
  test_expr_ops(1, vec1 * vec2);
  test_expr_ops(2, vec1 * vec3);
  test_expr_ops(6, vec3 * vec4);
}


void
test_tags()
{
  vsip::Vector<float> vec1(5);
  vsip::Vector<float> vec2(5);
  vsip::Vector<std::complex<float> > vec3(5);
  vsip::Vector<std::complex<float> > vec4(5);
  std::complex<double> z(2,1);

  // unary
  test_expr_tag("sin(S)", sin(vec1));
  test_expr_tag("exp(C)", exp(vec3));

  // binary
  test_expr_tag("max(S,S)", max(vec1, vec2));
  test_expr_tag("+(S,S)", vec1 + vec2);
  test_expr_tag("*(S,S)", vec1 * vec2);
  test_expr_tag("*(S,C)", vec1 * vec3);
  test_expr_tag("*(C,C)", vec3 * vec4);
  test_expr_tag("-(+(S,S),C)", (vec1 + vec2) - vec3);
  test_expr_tag("+(S,-(S,C))", vec1 + (vec2 - vec3));
  test_expr_tag("ma(S,s,/(C,z))", vec1 * 3.f + vec3 / z);

  // ternary
  test_expr_tag("expoavg(S,S,C)", expoavg(vec1, vec2, vec3));
}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_op_counts();
  test_tags();

  return 0;
}
