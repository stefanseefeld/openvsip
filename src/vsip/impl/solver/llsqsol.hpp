//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_solver_llsqsol_hpp_
#define vsip_impl_solver_llsqsol_hpp_

#include <vsip/matrix.hpp>
#include <vsip/impl/solver/qr.hpp>

#define VSIP_IMPL_USE_QRD_LSQSOL 1

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
llsqsol(Matrix<T, Block0>       a,
	const_Matrix<T, Block1> b,
	Matrix<T, Block2>       x)
  VSIP_THROW((std::bad_alloc, computation_error))
{
  length_type m = a.size(0);
  length_type n = a.size(1);


  // p = b.size(1)

  // b should be (m, p)
  OVXX_PRECONDITION(b.size(0) == m);
  OVXX_PRECONDITION(b.size(1) == b.size(1));
    
  // x should be (n, p)
  OVXX_PRECONDITION(x.size(0) == n);
  OVXX_PRECONDITION(x.size(1) == b.size(1));
    

  storage_type qrd_type;

  // Determine whether to use skinny or full QR.
  if (ovxx::qrd_traits<qrd<T, by_reference> >::supports_qrd_saveq1)
    qrd_type = qrd_saveq1;
  else if (ovxx::qrd_traits<qrd<T, by_reference> >::supports_qrd_saveq)
    qrd_type = qrd_saveq;
  else
    OVXX_DO_THROW(ovxx::unimplemented(
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
llsqsol(Matrix<T, Block0>       a,
	const_Matrix<T, Block1> b)
  VSIP_THROW((std::bad_alloc, computation_error))
{
  Matrix<T> x(a.size(1), b.size(1));
  llsqsol(a, b, x);
  return x;
}

} // namespace vsip

#endif
