/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/solver/covsol.hpp
    @author  Jules Bergmann
    @date    2005-08-23
    @brief   VSIPL++ Library: Covariance solver function.

*/

#ifndef VSIP_CORE_SOLVER_COVSOL_HPP
#define VSIP_CORE_SOLVER_COVSOL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/metaprogramming.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/solver/qr.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// Solver covariance system  A' A x = b, return x by reference.
///
/// Requires
///   A to be an M x N full-rank matrix
///   B to be an N x P matrix.
///   X to be an N x P matrix.

template <typename T,
	  typename Block0,
	  typename Block1,
	  typename Block2>
Matrix<T, Block2>
covsol(
  Matrix<T, Block0>       a,
  const_Matrix<T, Block1> b,
  Matrix<T, Block2>       x)
VSIP_THROW((std::bad_alloc, computation_error))
{
  length_type m = a.size(0);
  length_type n = a.size(1);
  length_type p = b.size(1);
  
  mat_op_type const tr = impl::is_complex<T>::value ? mat_herm : mat_trans;
  
  // b should be (n, p)
  assert(b.size(0) == n);
  assert(b.size(1) == p);
    
  // x should be (n, p)
  assert(x.size(0) == n);
  assert(x.size(1) == p);
    
  qrd<T, by_reference> qr(m, n, qrd_nosaveq);
    
  if (!qr.decompose(a))
    VSIP_IMPL_THROW(computation_error("covsol - qr.decompose failed"));
    
  Matrix<T> b_1(n, p);
    
  // 1: solve R' b_1 = b
  if (!qr.template rsol<tr>(b, T(1), b_1))
    VSIP_IMPL_THROW(computation_error("covsol - qr.rsol (1) failed"));
    
  // 2: solve R x = b_1 
  if (!qr.template rsol<mat_ntrans>(b_1, T(1), x))
    VSIP_IMPL_THROW(computation_error("covsol - qr.rsol (2) failed"));
    
  return x;
}



/// Solver covariance system  A' A x = b, return x by value.
///
/// Requires
///   A to be an M x N full-rank matrix
///   B to be an N x P matrix.
///   X to be an N x P matrix.

template <typename T,
	  typename Block0,
	  typename Block1>
Matrix<T>
covsol(
  Matrix<T, Block0>       a,
  const_Matrix<T, Block1> b)
{
  Matrix<T> x(b.size(0), b.size(1));
  covsol(a, b, x);
  return x;
}

} // namespace vsip

#endif // VSIP_CORE_SOLVER_COVSOL_HPP
