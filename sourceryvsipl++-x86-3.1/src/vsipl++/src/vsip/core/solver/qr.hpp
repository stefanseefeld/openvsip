/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved. */

#ifndef vsip_core_solver_qr_hpp_
#define vsip_core_solver_qr_hpp_

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/working_view.hpp>
#include <vsip/core/solver/common.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_LAPACK
#  include <vsip/opt/lapack/qr.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_SAL
#  include <vsip/opt/sal/qr.hpp>
#endif
#ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/cml/qr.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_CUDA
#  include <vsip/opt/cuda/qr.hpp>
#endif
#ifdef VSIP_IMPL_HAVE_CVSIP
#  include <vsip/core/cvsip/qr.hpp>
#endif

namespace vsip_csl
{
namespace dispatcher
{
#ifndef VSIP_IMPL_REF_IMPL
template <>
struct List<op::qrd>
{
  typedef Make_type_list<be::user,
			 be::cuda,
			 be::cml,
			 be::mercury_sal,
			 be::lapack>::type type;
};
#endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{
// Qrd traits.  Determine which QR types (no-save, skinny, full) are
// supported.
template <typename T>
struct Qrd_traits
{
  static bool const supports_qrd_saveq1  = T::backend_type::supports_qrd_saveq1;
  static bool const supports_qrd_saveq   = T::backend_type::supports_qrd_saveq;
  static bool const supports_qrd_nosaveq = T::backend_type::supports_qrd_nosaveq;
};

} // namespace vsip::impl

/// QR solver object.
template <typename = VSIP_DEFAULT_VALUE_TYPE,
	  return_mechanism_type ReturnMechanism = by_value>
class qrd;

// QR solver object (by-reference).
template <typename T>
class qrd<T, by_reference>
{
#ifndef VSIP_IMPL_REF_IMPL
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::qrd, T>::type
  backend_type;
#else
  typedef typename impl::cvsip::Qrd<T> backend_type;
#endif
  friend struct impl::Qrd_traits<qrd<T, by_reference> >;

public:
  qrd(length_type rows, length_type cols, storage_type st) 
  VSIP_THROW((std::bad_alloc))
  : backend_(rows, cols, st)
  {}

  length_type  rows()     const VSIP_NOTHROW { return backend_.rows();}
  length_type  columns()  const VSIP_NOTHROW { return backend_.columns();}
  storage_type qstorage() const VSIP_NOTHROW { return backend_.qstorage();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW { return backend_.decompose(m);}

  template <mat_op_type tr, product_side_type ps,
	    typename Block0, typename Block1>
  bool prodq(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.prodq<tr, ps>(b, x);}

  template <mat_op_type tr, typename Block0, typename Block1>
  bool rsol(const_Matrix<T, Block0> b, T const alpha, Matrix<T, Block1> x)
    VSIP_NOTHROW
  { return backend_.rsol<tr>(b, alpha, x);}

  template <typename Block0, typename Block1>
  bool covsol(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.covsol(b, x);}

  template <typename Block0, typename Block1>
  bool lsqsol(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  { return backend_.lsqsol(b, x);}

private:
  backend_type backend_;
};

// QR solver object (by-value).
template <typename T>
class qrd<T, by_value>
{
#ifndef VSIP_IMPL_REF_IMPL
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::qrd, T>::type
  backend_type;
#else
  typedef typename impl::cvsip::Qrd<T> backend_type;
#endif
  friend struct impl::Qrd_traits<qrd<T, by_value> >;

public:
  qrd(length_type rows, length_type cols, storage_type st) VSIP_THROW((std::bad_alloc))
    : backend_(rows, cols, st)
    {}

  length_type  rows()     const VSIP_NOTHROW { return backend_.rows();}
  length_type  columns()  const VSIP_NOTHROW { return backend_.columns();}
  storage_type qstorage() const VSIP_NOTHROW { return backend_.qstorage();}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW { return backend_.decompose(m);}

  template <mat_op_type tr, product_side_type ps, typename Block0>
  Matrix<T>
  prodq(const_Matrix<T, Block0> b) VSIP_NOTHROW
  {
    length_type x_rows, x_cols;
    if (ps == mat_lside)
    {
      x_rows = (qstorage() == qrd_saveq) ? rows() : 
	       (tr == mat_trans || tr == mat_herm) ? columns() : rows();
      x_cols = b.size(1);
    }
    else
    {
      x_rows = b.size(0);
      x_cols = (qstorage() == qrd_saveq) ? rows() : 
	       (tr == mat_trans || tr == mat_herm) ? rows() : columns();
    }

    Matrix<T> x(x_rows, x_cols);
    backend_.prodq<tr, ps>(b, x);
    return x;
  }

  template <mat_op_type tr, typename Block0>
  Matrix<T>
  rsol(const_Matrix<T, Block0> b, T const alpha) VSIP_NOTHROW
  {
    Matrix<T> x(b.size(0), b.size(1));
    backend_.rsol<tr>(b, alpha, x); 
    return x;
  }

  template <typename Block0>
  Matrix<T>
  covsol(const_Matrix<T, Block0> b) VSIP_NOTHROW
  {
    Matrix<T> x(b.size(0), b.size(1));
    backend_.covsol(b, x);
    return x;
  }

  template <typename Block0>
  Matrix<T>
  lsqsol(const_Matrix<T, Block0> b) VSIP_NOTHROW
  {
    Matrix<T> x(columns(), b.size(1));
    backend_.lsqsol(b, x);
    return x;
  }

private:
  backend_type backend_;
};

} // namespace vsip

#endif
