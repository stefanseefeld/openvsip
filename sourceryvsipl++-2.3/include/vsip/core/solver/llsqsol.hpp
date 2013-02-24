/* Copyright (c) 2005, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/solver/llsqsol.hpp
    @author  Jules Bergmann
    @date    2005-08-23
    @brief   VSIPL++ Library: Linear least-square solver function.

*/

#ifndef VSIP_CORE_SOLVER_LLSQSOL_HPP
#define VSIP_CORE_SOLVER_LLSQSOL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/metaprogramming.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/solver/qr.hpp>

#define VSIP_IMPL_USE_QRD_LSQSOL 1



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// Solver least-squares system min_x norm-2 (A x - b), return x by reference.
///
/// Requires
///   A to be an M x N full-rank matrix
///   B to be an M x P matrix.
///   X to be an N x P matrix.

template <typename T,
	  typename Block0,
	  typename Block1,
	  typename Block2>
Matrix<T, Block2>
llsqsol(
  Matrix<T, Block0>       a,
  const_Matrix<T, Block1> b,
  Matrix<T, Block2>       x)
VSIP_THROW((std::bad_alloc, computation_error))
{
  length_type m = a.size(0);
  length_type n = a.size(1);


  // p = b.size(1)

  // b should be (m, p)
  assert(b.size(0) == m);
  assert(b.size(1) == b.size(1));
    
  // x should be (n, p)
  assert(x.size(0) == n);
  assert(x.size(1) == b.size(1));
    

  storage_type qrd_type;

  // Determine whether to use skinny or full QR.
  if (impl::Qrd_traits<qrd<T, by_reference> >::supports_qrd_saveq1)
    qrd_type = qrd_saveq1;
  else if (impl::Qrd_traits<qrd<T, by_reference> >::supports_qrd_saveq)
    qrd_type = qrd_saveq;
  else
    VSIP_IMPL_THROW(impl::unimplemented(
	      "llsqsol: qrd supports neither qrd_saveq1 or qrd_saveq"));
  
  qrd<T, by_reference> qr(m, n, qrd_type);
  
  qr.decompose(a);
  
  qr.lsqsol(b, x);

  return x;
}



template <typename T,
	  typename Block0,
	  typename Block1>
Matrix<T>
llsqsol(
  Matrix<T, Block0>       a,
  const_Matrix<T, Block1> b)
VSIP_THROW((std::bad_alloc, computation_error))
{
  Matrix<T> x(a.size(1), b.size(1));
  llsqsol(a, b, x);
  return x;
}

} // namespace vsip

#endif // VSIP_CORE_SOLVER_LLSQSOL_HPP
