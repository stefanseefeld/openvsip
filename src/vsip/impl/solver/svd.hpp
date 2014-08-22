//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_solver_svd_hpp_
#define vsip_impl_solver_svd_hpp_

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/impl/math_enum.hpp>
#include <ovxx/dispatch.hpp>
#ifdef OVXX_HAVE_LAPACK
#  include <ovxx/lapack/svd.hpp>
#endif

namespace ovxx
{
namespace dispatcher
{
template <>
struct List<op::svd>
{
  typedef make_type_list<be::user,
			 be::lapack>::type type;
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{

/// SVD solver object.
template <typename = VSIP_DEFAULT_VALUE_TYPE,
	  return_mechanism_type R = by_value>
class svd;

template <typename T>
class svd<T, by_reference>
{
  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::svd, T>::type
  backend_type;

public:
  svd(length_type rows, length_type cols, storage_type ust, storage_type vst)
    VSIP_THROW((std::bad_alloc))
    : backend_(rows, cols, ust, vst)
  {}

  length_type rows() const VSIP_NOTHROW { return backend_.rows();}
  length_type columns() const VSIP_NOTHROW { return backend_.columns();}
  storage_type ustorage() const VSIP_NOTHROW { return backend_.ustorage();}
  storage_type vstorage() const VSIP_NOTHROW { return backend_.vstorage();}

  template <typename Block0, typename Block1>
  bool decompose(Matrix<T, Block0> m, Vector<scalar_f, Block1> dest) VSIP_NOTHROW
  { return backend_.decompose(m, dest);}

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool produ(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.template produ<tr, ps>(b, x);}

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool prodv(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.template prodv<tr, ps>(b, x);}

  template <typename Block>
  bool u(index_type low, index_type high, Matrix<T, Block> dest) VSIP_NOTHROW
  { return backend_.u(low, high, dest);}

  template <typename Block>
  bool v(index_type low, index_type high, Matrix<T, Block> dest) VSIP_NOTHROW
  { return backend_.v(low, high, dest);}

private:
  backend_type backend_;
};

template <typename T>
class svd<T, by_value>
{
  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::svd, T>::type
  backend_type;

public:
  svd(length_type rows, length_type cols, storage_type ust, storage_type vst)
    VSIP_THROW((std::bad_alloc))
    : backend_(rows, cols, ust, vst)
  {}
  length_type rows() const VSIP_NOTHROW { return backend_.rows();}
  length_type columns() const VSIP_NOTHROW { return backend_.columns();}
  storage_type ustorage() const VSIP_NOTHROW { return backend_.ustorage();}
  storage_type vstorage() const VSIP_NOTHROW { return backend_.vstorage();}
  template <typename Block0>
  Vector<scalar_f>
  decompose(Matrix<T, Block0> m) VSIP_THROW((std::bad_alloc, computation_error))
  {
    Vector<scalar_f> dest(backend_.order());
    if (!backend_.decompose(m, dest))
      OVXX_DO_THROW(computation_error("svd::decompose"));
    return dest;
  }

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block>
  Matrix<T>
  produ(const_Matrix<T, Block> b) VSIP_THROW((std::bad_alloc, computation_error))
  {
    length_type q_rows = rows();
    length_type q_cols = ustorage() == svd_uvfull ? rows() : backend_.order();

    length_type x_rows, x_cols;
    if (ps == mat_lside)
    {
      x_rows = (tr == mat_ntrans) ? q_rows : q_cols;
      x_cols = b.size(1);
    }
    else /* (ps == mat_rside) */
    {
      x_rows = b.size(0);
      x_cols = (tr == mat_ntrans) ? q_cols : q_rows;
    }
    Matrix<T> x(x_rows, x_cols);
    backend_.template produ<tr, ps>(b, x);
    return x;
  }

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block>
  Matrix<T>
  prodv(const_Matrix<T, Block> b) VSIP_THROW((std::bad_alloc, computation_error))
  { 
    length_type vt_rows = vstorage() == svd_uvfull ? columns() : backend_.order();
    length_type vt_cols = columns();

    length_type x_rows, x_cols;
    if (ps == mat_lside)
    {
      x_rows = (tr == mat_ntrans) ? vt_cols : vt_rows;
      x_cols = b.size(1);
    }
    else /* (ps == mat_rside) */
    {
      x_rows = b.size(0);
      x_cols = (tr == mat_ntrans) ? vt_rows : vt_cols;
    }
    Matrix<T> x(x_rows, x_cols);
    backend_.template prodv<tr, ps>(b, x);
    return x;
  }

  Matrix<T>
  u(index_type low, index_type high) VSIP_THROW((std::bad_alloc, computation_error))
  {
    OVXX_PRECONDITION((ustorage() == svd_uvpart && high <= backend_.order()) ||
		      (ustorage() == svd_uvfull && high <= rows()));

    Matrix<T> dest(rows(), high - low + 1);
    if (!backend_.u(low, high, dest)) OVXX_DO_THROW(computation_error("svd::u"));
    return dest;
  }

  Matrix<T>
  v(index_type low, index_type high) VSIP_THROW((std::bad_alloc, computation_error))
  {
    OVXX_PRECONDITION((vstorage() == svd_uvpart && high <= backend_.order()) ||
		      (vstorage() == svd_uvfull && high <= columns()));

    Matrix<T> dest(columns(), high - low + 1);
    if (!backend_.v(low, high, dest)) OVXX_DO_THROW(computation_error("svd::v"));
    return dest;
  }

private:
  backend_type backend_;
};

} // namespace vsip

#endif
