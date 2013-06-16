//
// Copyright (c) 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_qr_hpp_
#define ovxx_cvsip_qr_hpp_

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <ovxx/view_utils.hpp>
#include <vsip/impl/fns_elementwise.hpp>
#include <vsip/impl/solver/common.hpp>
#include <ovxx/cvsip/solver.hpp>
#include <ovxx/cvsip/view.hpp>

namespace ovxx
{
namespace cvsip
{

template <typename T>
class qrd
{
  typedef solver_traits<T> traits;
  static storage_format_type const storage_format = vsip::impl::dense_complex_format;
  typedef Layout<2, row2_type, dense, storage_format> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  static bool const supports_qrd_saveq1  = true;
  static bool const supports_qrd_saveq   = false;
  static bool const supports_qrd_nosaveq = true;

  qrd(length_type rows, length_type cols, storage_type st) VSIP_THROW((std::bad_alloc))
  : m_          (rows),
    n_          (cols),
    st_         (st),
    data_       (m_, n_),
    cvsip_data_ (data_.block().ptr(), m_, n_,true),
    qr_(traits::qr_create(m_, n_, st_))
  {
    OVXX_PRECONDITION(m_ > 0 && n_ > 0 && m_ >= n_);
    OVXX_PRECONDITION(st_ == qrd_nosaveq || st_ == qrd_saveq || st_ == qrd_saveq1);
  }
  qrd(qrd const &qr) VSIP_THROW((std::bad_alloc))
  : m_          (qr.m_),
    n_          (qr.n_),
    st_         (qr.st_),
    data_       (m_, n_),
    cvsip_data_ (data_.block().ptr(), m_, n_,true),
    qr_(traits::qr_create(m_, n_, st_))
  {
    data_ = qr.data_;
  }

  ~qrd() VSIP_NOTHROW { traits::qr_destroy(qr_);}

  length_type  rows()     const VSIP_NOTHROW { return m_;}
  length_type  columns()  const VSIP_NOTHROW { return n_;}
  storage_type qstorage() const VSIP_NOTHROW { return st_;}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(m.size(0) == m_ && m.size(1) == n_);

    cvsip_data_.block().release(false);
    assign_local(data_, m);
    cvsip_data_.block().admit(true);
    traits::qr_decompose(qr_, cvsip_data_.ptr());
    return true;
  }

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool prodq(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  template <mat_op_type       tr,
	    typename          Block0,
	    typename          Block1>
  bool rsol(const_Matrix<T, Block0>, T const, Matrix<T, Block1>) VSIP_NOTHROW;

  template <typename          Block0,
	    typename          Block1>
  bool covsol(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  template <typename          Block0,
	    typename          Block1>
  bool lsqsol(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

private:
  qrd& operator=(qrd const&) VSIP_NOTHROW;

  length_type  m_;			// Number of rows.
  length_type  n_;			// Number of cols.
  storage_type st_;			// Q storage type

  Matrix<T, data_block_type> data_;	// Factorized QR(mxn) matrix
  View<2,T,true>      cvsip_data_;
  typename traits::qr_type *qr_;
};

template <typename T>
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          Block0,
	  typename          Block1>
bool
qrd<T>::prodq(const_Matrix<T, Block0> b,
	      Matrix<T, Block1>       x) VSIP_NOTHROW
{
  typedef typename get_block_layout<Block0>::order_type order_type;
  static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
  typedef Layout<2, order_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  OVXX_PRECONDITION(this->qstorage() == qrd_saveq1 || this->qstorage() == qrd_saveq);
  length_type q_rows;
  length_type q_cols;
  if (qstorage() == qrd_saveq1)
  {
    q_rows = m_;
    q_cols = n_;
  }
  else // (qstorage() == qrd_saveq1)
  {
    q_rows = m_;
    q_cols = m_;
  }

  // do we need a transpose?
  if(tr == mat_trans || tr == mat_herm) 
  {
    std::swap(q_rows, q_cols);
  }
  if(tr == mat_herm) 
  {
    std::swap(q_rows, q_cols);
  }

  // left or right?
  if(ps == mat_lside) 
  {
    OVXX_PRECONDITION(b.size(0) == q_cols);
    OVXX_PRECONDITION(x.size(0) == q_rows);
    OVXX_PRECONDITION(b.size(1) == x.size(1));
  }
  else
  {
    OVXX_PRECONDITION(b.size(1) == q_rows);
    OVXX_PRECONDITION(x.size(1) == q_cols);
    OVXX_PRECONDITION(b.size(0) == x.size(0));
  }

  Matrix<T,block_type> b_int(b.size(0), b.size(1));
  dda::Data<block_type, dda::inout> b_data(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_data.ptr(),0,b_data.stride(0),b_data.size(0),
		  b_data.stride(1),b_data.size(1));

  cvsip_b_int.block().release(false);
  assign_local(b_int, b);
  cvsip_b_int.block().admit(true);
  int ret = traits::qr_prodq(qr_, tr, ps, cvsip_b_int.ptr());

  // now, copy into x
  cvsip_b_int.block().release(true);
  assign_local(x, b_int(Domain<2>(x.size(0),x.size(1))));
  return ret;
}

/// Solve op(R) x = alpha b

template <typename T>
template <mat_op_type tr,
	  typename    Block0,
	  typename    Block1>
bool
qrd<T>::rsol(const_Matrix<T, Block0> b, T alpha, Matrix<T, Block1> x) VSIP_NOTHROW
{
  typedef typename get_block_layout<Block0>::order_type order_type;
  static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
  typedef Layout<2, order_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  OVXX_PRECONDITION(b.size(0) == n_);
  OVXX_PRECONDITION(b.size(0) == x.size(0));
  OVXX_PRECONDITION(b.size(1) == x.size(1));

  Matrix<T, block_type> b_int(b.size(0), b.size(1));
  dda::Data<block_type, dda::inout>  b_data(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_data.ptr(),0,b_data.stride(0),b_data.size(0),
		  b_data.stride(1),b_data.size(1));

  cvsip_b_int.block().release(false);
  assign_local(b_int, b);
  cvsip_b_int.block().admit(true);

  traits::qr_solve_r(qr_, tr, alpha, cvsip_b_int.ptr());

  // copy b_int back into x
  cvsip_b_int.block().release(true);
  assign_local(x, b_int);
  return true;
}

/// Solve covariance system for x:
///   A' A X = B
template <typename T>
template <typename Block0, typename Block1>
bool
qrd<T>::covsol(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  typedef typename get_block_layout<Block0>::order_type order_type;
  static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
  typedef Layout<2, order_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  Matrix<T, block_type> b_int(b.size(0), b.size(1));
  dda::Data<block_type, dda::inout> b_data(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_data.ptr(),0,b_data.stride(0),b_data.size(0),
		  b_data.stride(1),b_data.size(1));

  cvsip_b_int.block().release(false);
  assign_local(b_int, b);
  cvsip_b_int.block().admit(true);

  traits::qr_solve(qr_, VSIP_COV, cvsip_b_int.ptr());

  // copy b_int back into x
  cvsip_b_int.block().release(true);
  assign_local(x, b_int);
  return true;
}

/// Solve linear least squares problem for x:
///   min_x norm-2( A x - b )
template <typename T>
template <typename Block0, typename Block1>
bool
qrd<T>::lsqsol(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  typedef typename get_block_layout<Block0>::order_type order_type;
  static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
  typedef Layout<2, order_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  Matrix<T, block_type> b_int(b.size(0), b.size(1));
  dda::Data<block_type, dda::inout> b_data(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_data.ptr(),0,b_data.stride(0),b_data.size(0),
		  b_data.stride(1),b_data.size(1));

  cvsip_b_int.block().release(false);
  assign_local(b_int, b);
  cvsip_b_int.block().admit(true);

  traits::qr_solve(qr_, VSIP_LLS, cvsip_b_int.ptr());
  // copy b_int back into x
  cvsip_b_int.block().release(true);
  assign_local(x, b_int(Domain<2>(x.size(0),x.size(1))));

  return true;
}

} // namespace ovxx::cvsip

namespace dispatcher
{
template <typename T>
struct Evaluator<op::qrd, be::cvsip, T>
{
  static bool const ct_valid = cvsip::solver_traits<T>::valid;
  typedef cvsip::qrd<T> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
