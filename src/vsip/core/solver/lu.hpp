//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_core_solver_lu_hpp_
#define vsip_core_solver_lu_hpp_

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/metaprogramming.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
#endif
#include <vsip/core/solver/common.hpp>
#ifdef VSIP_IMPL_HAVE_SAL
#  include <vsip/opt/sal/lu.hpp>
#endif
#ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/cml/lu.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_LAPACK
#  include <vsip/opt/lapack/lu.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_CVSIP
#  include <vsip/core/cvsip/lu.hpp>
#endif

namespace vsip_csl
{
namespace dispatcher
{
#ifndef VSIP_IMPL_REF_IMPL
template <>
struct List<op::lud>
{
  typedef Make_type_list<be::user,
			 be::cml,
			 be::mercury_sal,
			 be::lapack>::type type;
};
#endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
/// LU solver object.
template <typename = VSIP_DEFAULT_VALUE_TYPE,
	  return_mechanism_type ReturnMechanism = by_value>
class lud;

/// LU solver object (by-reference).
template <typename T>
class lud<T, by_reference>
{
#ifndef VSIP_IMPL_REF_IMPL
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::lud, T>::type
  backend_type;
#else
  typedef typename impl::cvsip::lu_solver<T> backend_type;
#endif

public:
  lud(length_type length) VSIP_THROW((std::bad_alloc)) : backend_(length) {}

  length_type length() const VSIP_NOTHROW { return backend_.length();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  { return backend_.decompose(m);}

  template <mat_op_type tr, typename Block0, typename Block1>
  bool solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.solve<tr>(b, x);}
private:
  backend_type backend_;
};

/// LU solver object (by-value).
template <typename T>
class lud<T, by_value>
{
#ifndef VSIP_IMPL_REF_IMPL
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::lud, T>::type
  backend_type;
#else
  typedef impl::cvsip::lu_solver<T> backend_type;
#endif

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
    backend_.solve<tr>(b, x); 
    return x;
  }

private:
  backend_type backend_;
};

} // namespace vsip


#endif
