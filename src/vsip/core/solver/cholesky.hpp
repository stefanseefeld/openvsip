//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_SOLVER_CHOLESKY_HPP
#define VSIP_CORE_SOLVER_CHOLESKY_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/solver/common.hpp>
#ifdef VSIP_IMPL_HAVE_SAL
#  include <vsip/opt/sal/cholesky.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_LAPACK
#  include <vsip/opt/lapack/cholesky.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_CVSIP
#  include <vsip/core/cvsip/cholesky.hpp>
#endif

namespace vsip_csl
{
namespace dispatcher
{
#ifndef VSIP_IMPL_REF_IMPL
template <>
struct List<op::chold>
{
  typedef Make_type_list<be::user,
			 be::mercury_sal,
			 be::lapack>::type type;
};
#endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{

/// CHOLESKY solver object.
template <typename = VSIP_DEFAULT_VALUE_TYPE,
	  return_mechanism_type ReturnMechanism = by_value>
class chold;

template <typename T>
class chold<T, by_reference>
{
#ifndef VSIP_IMPL_REF_IMPL
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::chold, T>::type
  backend_type;
#else
  typedef typename impl::cvsip::Chold<T> backend_type;
#endif

public:
  chold(mat_uplo uplo, length_type length) VSIP_THROW((std::bad_alloc))
  : backend_(uplo, length)
  {}

  length_type length()const VSIP_NOTHROW { return backend_.length();}
  mat_uplo    uplo()  const VSIP_NOTHROW { return backend_.uplo();}

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
#ifndef VSIP_IMPL_REF_IMPL
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::chold, T>::type
  backend_type;
#else
  typedef typename impl::cvsip::Chold<T> backend_type;
#endif

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

#endif // VSIP_OPT_SOLVER_CHOLESKY_HPP
