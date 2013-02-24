/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/expr-test.cpp
    @author  Jules Bergmann
    @date    2005-03-18
    @brief   VSIPL++ Library: tests for expression templates.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/test.hpp>
#include "block_interface.hpp"

using namespace std;
using namespace vsip;
using vsip_csl::equal;
namespace expr = vsip::impl::expr;

#ifndef ILLEGALCASE
#  define ILLEGALCASE 0
#endif



/***********************************************************************
  Definitions - Expression template functions.
***********************************************************************/

// Unary-negation test function.
//
// Requires:
//   V1 to be a vector.
// Returns:
//   View with expression block that computes element-wise negation of
//   V1.
template <typename T,
	  typename Block>
inline
const_Vector<T, expr::Unary<expr::op::Minus, Block, true> const>
t_neg(const_Vector<T, Block> v1)
{
  typedef expr::Unary<expr::op::Minus, Block, true> block_t;

  return const_Vector<T, const block_t>(block_t(v1.block()));
}



// Binary-addition test function.
//
// Requires:
//   V1, V2 to be element-conformant vectors.
// Returns:
//   View with block that computes element-wise addition of V1 and V2.
template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
inline
const_Vector<typename Promotion<T1, T2>::type,
	     expr::Binary<expr::op::Add, Block1, Block2, true> const>
t_add(const_Vector<T1, Block1> v1, const_Vector<T2, Block2> v2)
{
  typedef typename Promotion<T1, T2>::type RT;
  typedef expr::Binary<expr::op::Add, Block1, Block2, true> block_t;

  return const_Vector<RT, const block_t>(block_t(v1.block(), v2.block()));
}



// Binary-multiplication test function.
//
// Requires:
//   V1, V2 to be element-conformant vectors.
// Returns:
//   View with expression block that computes element-wise
//   multiplication of V1 and V2.
template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
inline
const_Vector<typename Promotion<T1, T2>::type,
	     expr::Binary<expr::op::Mult, Block1, Block2, true> const>
t_mul(const_Vector<T1, Block1> v1, const_Vector<T2, Block2> v2)
{
  typedef typename Promotion<T1, T2>::type RT;
  typedef expr::Binary<expr::op::Mult, Block1, Block2, true> block_t;

  return const_Vector<RT, const block_t>(block_t(v1.block(), v2.block()));
}



//
template <typename Block>
struct Block_name
{
  static string name() { return "*unknown*"; }
};



template <typename Block>
struct Block_name<const Block>
{
  static string name() { return string("const ") + Block_name<Block>::name(); }
};



template <dimension_type Dim,
	  typename       T,
	  typename       Order,
	  typename       Map>
struct Block_name<Dense<Dim, T, Order, Map> >
{
  static string name() { return string("Dense<>"); }
};



template <template <typename> class Operator, typename Block>
struct Block_name<expr::Unary<Operator, Block, true> >
{
  static string name() { return string("expr::Unary<")
      + Block_name<Block>::name() + ">";
  }
};



template <template <typename, typename> class Operator,
	  typename LBlock, typename RBlock>
struct Block_name<expr::Binary<Operator, LBlock, RBlock, true> >
{
  static string name() { return string("expr::Binary<")
      + Block_name<LBlock>::name() + ", "
      + Block_name<RBlock>::name() + ">";
  }
};



/***********************************************************************
  Definitions - Utility functions.
***********************************************************************/

template <typename T,
	  typename Block>
T
sum(const_Vector<T, Block> v)
{
  // std::cout << "sum(" << Block_name<Block>::name() << ")\n";
  T total = T();
  for (index_type i=0; i<v.length(); ++i)
    total += v.get(i);
  return total;
}



template <typename T,
	  typename Block>
T
vector_sum(Vector<T, Block> v)
{
  // std::cout << "vector_sum(" << Block_name<Block>::name() << ")\n";
  T total = T();
  for (index_type i=0; i<v.length(); ++i)
    total += v.get(i);
  return total;
}



/***********************************************************************
  Definitions - Tests.
***********************************************************************/



// Test unary-function expression template.
void
test_neg()
{
  length_type const len = 10;
  const_Vector<float> v1(len, 1.f);
  Vector<float> v2(len);

  v2 = t_neg(v1);

  test_assert(v2.get(0) == -1.f);
}

  

// Test binary- and unary- expression templates.
template <typename T>
void
test_expr()
{
  length_type const len = 10;
  Vector<T> v1(len, T());
  Vector<T> v2(len, T());
  Vector<T> v3(len, T());

  Vector<T> z1(len);

  v1.put(1, T(1));
  v2.put(1, T(2));
  v3.put(1, T(3));

  z1           = t_add(v1, t_add(v2, v3));
  Vector<T> z2 = t_add(t_neg(v1), t_add(v2, v3));
  Vector<T> z3 = t_add(v1, t_neg(t_add(v2, v3)));
  Vector<T> z4 = t_add(t_mul(v2, v3), t_neg(t_add(v1, v3)));

  Vector<T> z5(len);
  z5           = t_mul(v2, v3);

  test_assert(equal(z1.get(1), T(6)));
  test_assert(equal(z2.get(1), T(4)));
  test_assert(equal(z3.get(1), T(-4)));
  test_assert(equal(z4.get(1), T(2)));

}



// Test use of expression templates as function arguments.
template <typename T>
void
test_funcall()
{
  length_type const len = 10;
  Vector<T> v1(len, T());
  Vector<T> v2(len, T());
  Vector<T> v3(len, T());

  v1.put(1, T(1));
  v2.put(1, T(2));
  v3.put(1, T(3));

  T s1 = sum(v1);

  T s2 = sum(t_add(v1, v2));

  const_Vector<T, expr::Binary<expr::op::Add, Dense<1, T>, Dense<1, T>, true> const> vx(t_add(v1, v2));
  T s3 = sum(vx);

  T s4 = sum(t_add(t_mul(v2, v3), t_neg(t_add(v1, v3))));

#if ILLEGALCASE == 1
  // It should not be possible to pass an expression template (which is
  // a const_Vector) to a function expecting a Vector.
  T s5 = vector_sum(t_add(t_mul(v2, v3), t_neg(t_add(v1, v3))));
#endif

  test_assert(equal(s1, T(1)));
  test_assert(equal(s2, T(3)));
  test_assert(equal(s3, T(3)));
  test_assert(equal(s4, T(2)));
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_neg();
  test_expr<float>();
  test_expr<int>();
  test_expr<std::complex<float> >();
  test_funcall<float>();
  test_funcall<int>();
}
