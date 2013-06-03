//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_solver_cholesky_hpp_
#define vsip_impl_solver_cholesky_hpp_

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <vsip/impl/solver/common.hpp>
#ifdef OVXX_HAVE_LAPACK
#  include <ovxx/lapack/cholesky.hpp>
#endif
#ifdef OVXX_HAVE_CVSIP
#  include <ovxx/cvsip/cholesky.hpp>
#endif

namespace ovxx
{
namespace dispatcher
{
template <>
struct List<op::chold>
{
  typedef make_type_list<be::user,
			 be::cvsip,
			 be::lapack>::type type;
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{

/// Cholesky solver object.
template <typename = VSIP_DEFAULT_VALUE_TYPE,
	  return_mechanism_type R = by_value>
class chold;

template <typename T>
class chold<T, by_reference>
{
  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::chold, T>::type
  backend_type;

public:
  chold(mat_uplo uplo, length_type length) VSIP_THROW((std::bad_alloc))
  : backend_(uplo, length)
  {}

  length_type length() const VSIP_NOTHROW { return backend_.length();}
  mat_uplo uplo() const VSIP_NOTHROW { return backend_.uplo();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW { return backend_.decompose(m);}

  template <typename Block0, typename Block1>
  bool solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.solve(b, x);}

private:
  backend_type backend_;
};

template <typename T>
class chold<T, by_value>
{
  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::chold, T>::type
  backend_type;

public:
  chold(mat_uplo uplo, length_type length) VSIP_THROW((std::bad_alloc))
  : backend_(uplo, length)
  {}

  length_type length()const VSIP_NOTHROW { return backend_.length();}
  mat_uplo    uplo()  const VSIP_NOTHROW { return backend_.uplo();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW { return backend_.decompose(m);}

  template <typename Block0>
  Matrix<T>
  solve(const_Matrix<T, Block0> b) VSIP_NOTHROW
  {
    Matrix<T> x(b.size(0), b.size(1));
    backend_.solve(b, x); 
    return x;
  }

private:
  backend_type backend_;
};

} // namespace vsip

#endif
