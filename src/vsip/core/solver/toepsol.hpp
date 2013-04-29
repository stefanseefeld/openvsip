//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_SOLVER_TOEPSOL_HPP
#define VSIP_CORE_SOLVER_TOEPSOL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// Solve a real symmetric or complex Hermetian positive definite Toeplitz
/// linear system.
///
/// Solves:
///   :equation:`T x = B` for x, given T and B.
///
/// Arguments:
///   :t: an input vector of length N, first row of the Toeplitz matrix.
///   :b: an input vector of length N.
///   :y: a vector of length N used for temporary workspace.
///   :x: a vector of length N, for the result to be stored.
///
/// Effects:
///   On return, X contains solution to system :equation:`T X = B`.
///
/// Returns: X.
///
/// Throws:
///   computation_error if T is ill-formed (non postive definite)
///   bad_alloc if allocation fails.
///
template <typename T,
	  typename Block0,
	  typename Block1,
	  typename Block2,
	  typename Block3>
const_Vector<T, Block3>
toepsol(const_Vector<T, Block0> t,
	const_Vector<T, Block1> b,
	Vector<T, Block2>       y,
	Vector<T, Block3>       x)
VSIP_THROW((std::bad_alloc, computation_error))
{
  typedef typename impl::scalar_of<T>::type scalar_type;

  assert(t.size() == b.size());
  assert(t.size() == y.size());
  assert(t.size() == x.size());

  length_type n = t.size();


  typename const_Vector<T, Block0>::subview_type r = t(Domain<1>(1, 1, n-1));
  Vector<T> tmpv(n);

  scalar_type beta  = 1.0;
  scalar_type scale = impl::fn::impl_real(t.get(0));
  T           alpha = impl::fn::impl_conj(-r.get(0)/scale);
  T           tmps; 
  
  alpha = impl::fn::impl_conj(-r.get(0)/scale);
  
  y.put(0, alpha);
  x.put(0, b.get(0) / scale);

  for (index_type k=1; k<n; ++k)
  {
    beta *= (1.0 - magsq(alpha));
    if (beta == 0.0)
    { 
      VSIP_IMPL_THROW(computation_error("TOEPSOL: not full rank"));
    } 

    tmps = dot(impl_conj(r(Domain<1>(k))), x(Domain<1>(k-1, -1, k)));
    T mu = (b.get(k) - tmps) / (scale*beta);

    // x(Domain<1>(k)) += mu * impl_conj(y(Domain<1>(k-1, -1, k)));
    x(Domain<1>(k)) = mu * impl_conj(y(Domain<1>(k-1, -1, k)))
                    + x(Domain<1>(k));
    x.put(k, mu);

    if (k < (n - 1))
    {
      tmps  = dot(impl_conj(r(Domain<1>(k))), y(Domain<1>(k-1, -1, k)));
      alpha = -(tmps + impl::fn::impl_conj(r.get(k))) / (scale*beta);
      
      tmpv(Domain<1>(k)) = alpha * impl_conj(y(Domain<1>(k-1, -1, k))) 
	                 + y(Domain<1>(k));
      y(Domain<1>(k)) = tmpv(Domain<1>(k));
      y.put(k, alpha);
    }
  }

  return x;
}



/// Solve a real symmetric or complex Hermetian positive definite Toeplitz
/// linear system.

/// Solves:
///   T x = B
/// for x, given T and B.
///
/// Requires:
///   T to be an input vector of length N, first row of the Toeplitz matrix.
///   B to be an input vector of length N.
///   Y to be a vector of length N used for temporary workspace.
///
/// Effects:
///   Returns vector X of length N, solution to system T X = B.
///
/// Throws:
///   computation_error if T is ill-formed (non postive definite)
///   bad_alloc if allocation fails.

template <typename T,
	  typename Block0,
	  typename Block1,
	  typename Block2>
const_Vector<T>
toepsol(
   const_Vector<T, Block0> t,
   const_Vector<T, Block1> b,
   Vector<T, Block2>       y)
VSIP_THROW((std::bad_alloc, computation_error))
{
  Vector<T> x(t.size());
  return toepsol(t, b, y, x);
}

} // namespace vsip

#endif // VSIP_CORE_SOLVER_TOEPSOL_HPP
