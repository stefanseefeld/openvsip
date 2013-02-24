/* Copyright (c) 2006, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/cvsip/qr.hpp
    @author  Assem Salama
    @date    2006-10-26
    @brief   VSIPL++ Library: QR linear system solver using CVSIP.

*/

#ifndef VSIP_CORE_CVSIP_SOLVER_QR_HPP
#define VSIP_CORE_CVSIP_SOLVER_QR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/working_view.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/core/solver/common.hpp>
#include <vsip/core/cvsip/solver.hpp>
#include <vsip/core/cvsip/view.hpp>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cvsip
{

/// Qrd implementation using CVSIP

/// Requires:
///   T to be a value type supported by SAL's QR routines
template <typename T>
class Qrd
{
  typedef Solver_traits<T> traits;
  typedef vsip::impl::dense_complex_type   complex_type;
  typedef Layout<2, row2_type, Stride_unit_dense, complex_type> data_LP;
  typedef Strided<2, T, data_LP> data_block_type;

  static bool const supports_qrd_saveq1  = true;
  static bool const supports_qrd_saveq   = false;
  static bool const supports_qrd_nosaveq = true;

public:
  Qrd(length_type rows, length_type cols, storage_type st) VSIP_THROW((std::bad_alloc))
  : m_          (rows),
    n_          (cols),
    st_         (st),
    data_       (m_, n_),
    cvsip_data_ (data_.block().impl_data(), m_, n_,true),
    qr_(traits::qr_create(m_, n_, st_))
  {
    assert(m_ > 0 && n_ > 0 && m_ >= n_);
    assert(st_ == qrd_nosaveq || st_ == qrd_saveq || st_ == qrd_saveq1);
  }
  Qrd(Qrd const &qr) VSIP_THROW((std::bad_alloc))
  : m_          (qr.m_),
    n_          (qr.n_),
    st_         (qr.st_),
    data_       (m_, n_),
    cvsip_data_ (data_.block().impl_data(), m_, n_,true),
    qr_(traits::qr_create(m_, n_, st_))
  {
    data_ = qr.data_;
  }

  ~Qrd() VSIP_NOTHROW { traits::qr_destroy(qr_);}

  length_type  rows()     const VSIP_NOTHROW { return m_; }
  length_type  columns()  const VSIP_NOTHROW { return n_; }
  storage_type qstorage() const VSIP_NOTHROW { return st_; }

  /// Decompose matrix M into QR form.
  ///
  /// Requires
  ///   M to be a full rank, modifiable matrix of ROWS x COLS.
  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    assert(m.size(0) == m_ && m.size(1) == n_);

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
  typedef std::vector<T, Aligned_allocator<T> > vector_type;

  Qrd& operator=(Qrd const&) VSIP_NOTHROW;

  length_type  m_;			// Number of rows.
  length_type  n_;			// Number of cols.
  storage_type st_;			// Q storage type

  Matrix<T, data_block_type> data_;	// Factorized QR(mxn) matrix
  View<2,T,true>      cvsip_data_;
  typename traits::qr_type *qr_;
};

/// Compute product of Q and b
/// 
/// If qstorage == qrd_saveq1, Q is MxN.
/// If qstorage == qrd_saveq,  Q is MxM.
///
/// qstoarge   | ps        | tr         | product | b (in) | x (out)
/// qrd_saveq1 | mat_lside | mat_ntrans | Q b     | (n, p) | (m, p)
/// qrd_saveq1 | mat_lside | mat_trans  | Q' b    | (m, p) | (n, p)
/// qrd_saveq1 | mat_lside | mat_herm   | Q* b    | (m, p) | (n, p)
///
/// qrd_saveq1 | mat_rside | mat_ntrans | b Q     | (p, m) | (p, n)
/// qrd_saveq1 | mat_rside | mat_trans  | b Q'    | (p, n) | (p, m)
/// qrd_saveq1 | mat_rside | mat_herm   | b Q*    | (p, n) | (p, m)
///
/// qrd_saveq  | mat_lside | mat_ntrans | Q b     | (m, p) | (m, p)
/// qrd_saveq  | mat_lside | mat_trans  | Q' b    | (m, p) | (m, p)
/// qrd_saveq  | mat_lside | mat_herm   | Q* b    | (m, p) | (m, p)
///
/// qrd_saveq  | mat_rside | mat_ntrans | b Q     | (p, m) | (p, m)
/// qrd_saveq  | mat_rside | mat_trans  | b Q'    | (p, m) | (p, m)
/// qrd_saveq  | mat_rside | mat_herm   | b Q*    | (p, m) | (p, m)
template <typename T>
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          Block0,
	  typename          Block1>
bool
Qrd<T>::prodq(const_Matrix<T, Block0> b,
	      Matrix<T, Block1>       x) VSIP_NOTHROW
{
  typedef typename Block_layout<Block0>::order_type order_type;
  typedef typename Block_layout<Block0>::complex_type complex_type;
  typedef Layout<2, order_type, Stride_unit_dense, complex_type> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  assert(this->qstorage() == qrd_saveq1 || this->qstorage() == qrd_saveq);
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
    assert(b.size(0) == q_cols);
    assert(x.size(0) == q_rows);
    assert(b.size(1) == x.size(1));
  }
  else
  {
    assert(b.size(1) == q_rows);
    assert(x.size(1) == q_cols);
    assert(b.size(0) == x.size(0));
  }

  Matrix<T,block_type> b_int(b.size(0), b.size(1));
  Ext_data<block_type> b_ext(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_ext.data(),0,b_ext.stride(0),b_ext.size(0),
		                 b_ext.stride(1),b_ext.size(1));

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
Qrd<T>::rsol(const_Matrix<T, Block0> b, T alpha, Matrix<T, Block1> x) VSIP_NOTHROW
{
  typedef typename Block_layout<Block0>::order_type order_type;
  typedef typename Block_layout<Block0>::complex_type complex_type;
  typedef Layout<2, order_type, Stride_unit_dense, complex_type> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  assert(b.size(0) == n_);
  assert(b.size(0) == x.size(0));
  assert(b.size(1) == x.size(1));

  Matrix<T, block_type> b_int(b.size(0), b.size(1));
  Ext_data<block_type>  b_ext(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_ext.data(),0,b_ext.stride(0),b_ext.size(0),
		                 b_ext.stride(1),b_ext.size(1));

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
Qrd<T>::covsol(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  typedef typename Block_layout<Block0>::order_type order_type;
  typedef typename Block_layout<Block0>::complex_type complex_type;
  typedef Layout<2, order_type, Stride_unit_dense, complex_type> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  Matrix<T, block_type> b_int(b.size(0), b.size(1));
  Ext_data<block_type>  b_ext(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_ext.data(),0,b_ext.stride(0),b_ext.size(0),
		                 b_ext.stride(1),b_ext.size(1));

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
Qrd<T>::lsqsol(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  typedef typename Block_layout<Block0>::order_type order_type;
  typedef typename Block_layout<Block0>::complex_type complex_type;
  typedef Layout<2, order_type, Stride_unit_dense, complex_type> data_LP;
  typedef Strided<2, T, data_LP, Local_map> block_type;

  Matrix<T, block_type> b_int(b.size(0), b.size(1));
  Ext_data<block_type> b_ext(b_int.block());
  cvsip::View<2,T,true>
      cvsip_b_int(b_ext.data(),0,b_ext.stride(0),b_ext.size(0),
		                 b_ext.stride(1),b_ext.size(1));

  cvsip_b_int.block().release(false);
  assign_local(b_int, b);
  cvsip_b_int.block().admit(true);

  traits::qr_solve(qr_, VSIP_LLS, cvsip_b_int.ptr());
  // copy b_int back into x
  cvsip_b_int.block().release(true);
  assign_local(x, b_int(Domain<2>(x.size(0),x.size(1))));

  return true;
}

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T>
struct Evaluator<op::qrd, be::cvsip, T>
{
  static bool const ct_valid = impl::cvsip::Solver_traits<T>::valid;
  typedef impl::cvsip::Qrd<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_CORE_CVSIP_SOLVER_QR_HPP
