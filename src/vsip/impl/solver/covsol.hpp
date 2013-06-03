//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_solver_covsol_hpp_
#define vsip_impl_solver_covsol_hpp_

#include <vsip/matrix.hpp>
#include <vsip/impl/solver/qr.hpp>

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
covsol(Matrix<T, Block0>       a,
       const_Matrix<T, Block1> b,
       Matrix<T, Block2>       x)
  VSIP_THROW((std::bad_alloc, computation_error))
{
  length_type m = a.size(0);
  length_type n = a.size(1);
  length_type p = b.size(1);
  
  mat_op_type const tr = ovxx::is_complex<T>::value ? mat_herm : mat_trans;
  
  // b should be (n, p)
  OVXX_PRECONDITION(b.size(0) == n);
  OVXX_PRECONDITION(b.size(1) == p);
    
  // x should be (n, p)
  OVXX_PRECONDITION(x.size(0) == n);
  OVXX_PRECONDITION(x.size(1) == p);
    
  qrd<T, by_reference> qr(m, n, qrd_nosaveq);
    
  if (!qr.decompose(a))
    OVXX_DO_THROW(computation_error("covsol - qr.decompose failed"));
    
  Matrix<T> b_1(n, p);
    
  // 1: solve R' b_1 = b
  if (!qr.template rsol<tr>(b, T(1), b_1))
    OVXX_DO_THROW(computation_error("covsol - qr.rsol (1) failed"));
    
  // 2: solve R x = b_1 
  if (!qr.template rsol<mat_ntrans>(b_1, T(1), x))
    OVXX_DO_THROW(computation_error("covsol - qr.rsol (2) failed"));
    
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
covsol(Matrix<T, Block0>       a,
       const_Matrix<T, Block1> b)
{
  Matrix<T> x(b.size(0), b.size(1));
  covsol(a, b, x);
  return x;
}

} // namespace vsip

#endif
