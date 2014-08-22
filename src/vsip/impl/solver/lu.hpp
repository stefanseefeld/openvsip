//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_solver_lu_hpp_
#define vsip_impl_solver_lu_hpp_

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <vsip/impl/solver/common.hpp>
#include <ovxx/dispatch.hpp>
#ifdef OVXX_HAVE_LAPACK
#  include <ovxx/lapack/lu.hpp>
#endif
#ifdef OVXX_HAVE_CVSIP
#  include <ovxx/cvsip/lu.hpp>
#endif

namespace ovxx
{
namespace dispatcher
{
template <>
struct List<op::lud>
{
  typedef make_type_list<be::user,
			 be::cvsip,
			 be::lapack>::type type;
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{
/// LU solver object.
template <typename = VSIP_DEFAULT_VALUE_TYPE,
	  return_mechanism_type R = by_value>
class lud;

/// LU solver object (by-reference).
template <typename T>
class lud<T, by_reference>
{
  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::lud, T>::type
  backend_type;

public:
  lud(length_type length) VSIP_THROW((std::bad_alloc)) : backend_(length) {}

  length_type length() const VSIP_NOTHROW { return backend_.length();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  { return backend_.decompose(m);}

  template <mat_op_type tr, typename Block0, typename Block1>
  bool solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.template solve<tr>(b, x);}
private:
  backend_type backend_;
};

/// LU solver object (by-value).
template <typename T>
class lud<T, by_value>
{
  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::lud, T>::type
  backend_type;

public:
  lud(length_type length) VSIP_THROW((std::bad_alloc)) : backend_(length) {}

  length_type length() const VSIP_NOTHROW { return backend_.length();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  { return backend_.decompose(m);}

  template <mat_op_type tr, typename Block0>
  Matrix<T> solve(const_Matrix<T, Block0> b) VSIP_NOTHROW
  {
    Matrix<T> x(b.size(0), b.size(1));
    backend_.template solve<tr>(b, x); 
    return x;
  }

private:
  backend_type backend_;
};

} // namespace vsip


#endif
