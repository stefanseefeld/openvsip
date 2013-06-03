//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_qr_hpp_
#define ovxx_lapack_qr_hpp_

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <ovxx/lapack/blas.hpp>
#include <ovxx/lapack/lapack.hpp>
#include <ovxx/aligned_array.hpp>
#include <vsip/impl/solver/common.hpp>

namespace ovxx
{
namespace lapack
{

template <typename T, bool Blocked>
class qrd : ct_assert<blas::traits<T>::valid>
{
  typedef Layout<2, col2_type, dense, array> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  static bool const supports_qrd_saveq1  = true;
  static bool const supports_qrd_saveq   = true;
  static bool const supports_qrd_nosaveq = true;

  qrd(length_type rows, length_type cols, storage_type s)
    VSIP_THROW((std::bad_alloc))
    : rows_(rows),
      cols_(cols),
      storage_(s),
      data_(rows_, cols_),
      tau_(cols_),
      work_(cols_ * lapack::geqrf_work<T>(rows_, cols_))
  {
    OVXX_PRECONDITION(rows_ > 0 && cols_ > 0 && rows_ >= cols_);
    OVXX_PRECONDITION(storage_ == qrd_nosaveq ||
	   storage_ == qrd_saveq ||
	   storage_ == qrd_saveq1);
  }

  qrd(qrd const &other) VSIP_THROW((std::bad_alloc))
    : rows_(other.rows_),
      cols_(other.cols_),
      storage_(other.storage_),
      data_(rows_, cols_),
      tau_(cols_),
      work_(cols_ * lapack::geqrf_work<T>(rows_, cols_))
  {
    data_ = other.data_;
    for (index_type i=0; i<cols_; ++i)
      tau_[i] = other.tau_[i];
    for (index_type i = 0; i != work_.size(); ++i)
      work_[i] = other.work_[i];
  }

  qrd &operator=(qrd const &other) VSIP_THROW((std::bad_alloc))
  {
    // TODO: At present assignment requires dimensions to match,
    //       as views aren't resizable.
    OVXX_PRECONDITION(rows_ == other.rows_ && cols_ == other.cols_);
    storage_ = other.storage_;
    data_ = other.data_;
    for (index_type i=0; i<cols_; ++i)
      tau_[i] = other.tau_[i];
    for (index_type i = 0; i != work_.size(); ++i)
      work_[i] = other.work_[i];
  }

  length_type rows() const VSIP_NOTHROW { return rows_;}
  length_type columns() const VSIP_NOTHROW { return cols_;}
  storage_type qstorage() const VSIP_NOTHROW { return storage_;}

  template <typename B>
  bool decompose(Matrix<T, B> m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(m.size(0) == rows_ && m.size(1) == cols_);
    int lwork = work_.size();
    parallel::assign_local(data_, m);
    dda::Data<data_block_type, dda::inout> data(data_.block());
    if (Blocked)
      lapack::geqrf(rows_, cols_, data.ptr(), rows_, tau_.get(),
		    work_.get(), lwork);
    else
      lapack::geqr2(rows_, cols_, data.ptr(), rows_, tau_.get(),
		    work_.get(), lwork);
    OVXX_PRECONDITION((length_type)lwork <= work_.size());
    return true;
  }

  template <mat_op_type tr, product_side_type ps,
	    typename B0, typename B1>
  bool prodq(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(this->qstorage() == qrd_saveq1 || this->qstorage() == qrd_saveq);

    char        side;
    char        trans;
    length_type q_rows;
    length_type q_cols;
    length_type k_reflectors = cols_;
    int         mqr_lwork;

    if (qstorage() == qrd_saveq1)
    {
      q_rows = rows_;
      q_cols = cols_;
    }
    else // (qstorage() == qrd_saveq1)
    {
      q_rows = rows_;
      q_cols = rows_;
    }

    if (tr == mat_trans)
    {
      trans = 't';
      std::swap(q_rows, q_cols);
    }
    else if (tr == mat_herm)
    {
      trans = 'c';
      std::swap(q_rows, q_cols);
    }
    else // if (tr == mat_ntrans)
    {
      trans = 'n';
    }
  
    if (ps == mat_lside)
    {
      OVXX_PRECONDITION(b.size(0) == q_cols);
      OVXX_PRECONDITION(x.size(0) == q_rows);
      OVXX_PRECONDITION(b.size(1) == x.size(1));
      side = 'l';
      mqr_lwork = b.size(1);
    }
    else // (ps == mat_rside)
    {
      OVXX_PRECONDITION(b.size(1) == q_rows);
      OVXX_PRECONDITION(x.size(1) == q_cols);
      OVXX_PRECONDITION(b.size(0) == x.size(0));
      side = 'r';
      mqr_lwork = b.size(0);
    }

    Matrix<T, data_block_type> b_int(b.size(0), b.size(1));
    parallel::assign_local(b_int, b);

    int work = lapack::mqr_work<T>(side, trans,
				   b.size(0), b.size(1), k_reflectors);
    mqr_lwork *= work;
    aligned_array<T> mqr_work(mqr_lwork);
    {
      dda::Data<data_block_type, dda::inout> b_data(b_int.block());
      dda::Data<data_block_type, dda::in> a_data(data_.block());
      lapack::mqr(side,
		  trans,
		  b.size(0), b.size(1),
		  k_reflectors,
		  a_data.ptr(), rows_,
		  tau_.get(),
		  b_data.ptr(), b.size(0),
		  mqr_work.get(), mqr_lwork);
		
    }
    parallel::assign_local(x, b_int);
    return true;
  }

  template <mat_op_type tr, typename B0, typename B1>
  bool rsol(const_Matrix<T, B0> b, T alpha, Matrix<T, B1> x)
    VSIP_NOTHROW
  {
    OVXX_PRECONDITION(b.size(0) == cols_);
    OVXX_PRECONDITION(b.size(0) == x.size(0));
    OVXX_PRECONDITION(b.size(1) == x.size(1));

    char trans;

    switch(tr)
    {
      case mat_trans:
	// assert(is_scalar<T>::value);
	trans = 't';
	break;
      case mat_herm:
	// assert(is_complex<T>::value);
	trans = 'c';
	break;
      default:
	trans = 'n';
	break;
    }

    Matrix<T, data_block_type> b_int(b.size(0), b.size(1));
    parallel::assign_local(b_int, b);
    {
      dda::Data<data_block_type, dda::in> a_data(data_.block());
      dda::Data<data_block_type, dda::inout> b_data(b_int.block());
      
      blas::trsm('l',		// R appears on [l]eft-side
		 'u',		// R is [u]pper-triangular
		 trans,		// 
		 'n',		// R is [n]ot unit triangular
		 b.size(0), b.size(1),
		 alpha,
		 a_data.ptr(), rows_,
		 b_data.ptr(), b_data.stride(1));
    }
    parallel::assign_local(x, b_int);
    return true;
  }

  template <typename B0, typename B1>
  bool covsol(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    length_type b_rows = b.size(0);
    length_type b_cols = b.size(1);
    T alpha = T(1);
    OVXX_PRECONDITION(b_rows == cols_);

    // Solve A' A x = b

    // Equiv to solve: R' R x = b
    // First solve:    R' b_1 = b
    // Then solve:     R x = b_1

    Matrix<T, data_block_type> b_int(b_rows, b_cols);
    parallel::assign_local(b_int, b);
    {
      dda::Data<data_block_type, dda::inout> b_data(b_int.block());
      dda::Data<data_block_type, dda::in> a_data(data_.block());

      // First solve: R' b_1 = b

      blas::trsm('l',	// R' appears on [l]eft-side
		 'u',	// R is [u]pper-triangular
		 blas::traits<T>::trans, // [c]onj/[t]ranspose (conj(R'))
		 'n',	// R is [n]ot unit triangular
		 b_rows, b_cols,
		 alpha,
		 a_data.ptr(), rows_,
		 b_data.ptr(), b_rows);

      // Then solve: R x = b_1
    
      blas::trsm('l',	// R appears on [l]eft-side
		 'u',	// R is [u]pper-triangular
		 'n',	// [n]o-op (R)
		 'n',	// R is [n]ot unit triangular
		 b_rows, b_cols,
		 alpha,
		 a_data.ptr(), rows_,
		 b_data.ptr(), b_rows);
    }
    parallel::assign_local(x, b_int);
    return true;
  }

  template <typename B0, typename B1>
  bool lsqsol(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    length_type p = b.size(1);
    OVXX_PRECONDITION(b.size(0) == rows_);
    OVXX_PRECONDITION(x.size(0) == cols_);
    OVXX_PRECONDITION(x.size(1) == p);

    length_type c_rows = rows_;
    length_type c_cols = p;
  
    int work = lapack::mqr_work<T>('l', blas::traits<T>::trans,
				   c_rows, c_cols, cols_);
    int mqr_lwork = c_cols*work;
    aligned_array<T> mqr_work(mqr_lwork);

    // Solve  A X = B  for X
    //
    // 0. factor:             QR X = B
    //    mult by Q'        Q'QR X = Q'B
    //    simplify             R X = Q'B
    //
    // 1. compute C = Q'B:     R X = C
    // 2. solve for X:         R X = C
    Matrix<T, data_block_type> c(c_rows, c_cols);
    parallel::assign_local(c, b);
    {
      dda::Data<data_block_type, dda::inout> c_data(c.block());
      dda::Data<data_block_type, dda::in> a_data(data_.block());

      // 1. compute C = Q'B:     R X = C

      lapack::mqr('l',				// Q' on [l]eft (C = Q' B)
		  blas::traits<T>::trans,	// [t]ranspose (Q')
		  c_rows, c_cols, 
		  cols_,			// No. elementary reflectors in Q
		  a_data.ptr(), rows_,
		  tau_.get(),
		  c_data.ptr(), c_rows,
		  mqr_work.get(), mqr_lwork);
		
      // 2. solve for X:         R X = C
      //      R (n, n)
      //      X (n, p)
      //      C (m, p)
      // Since R is (n, n), we treat C as an (n, p) matrix.

      blas::trsm('l',	// R appears on [l]eft-side
		 'u',	// R is [u]pper-triangular
		 'n',	// [n]o op (R)
		 'n',	// R is [n]ot unit triangular
		 cols_, c_cols,
		 T(1),
		 a_data.ptr(), rows_,
		 c_data.ptr(), c_rows);
    }
    parallel::assign_local(x, c(Domain<2>(cols_, p)));
    return true;
  }

private:
  length_type  rows_;			// Number of rows.
  length_type  cols_;			// Number of cols.
  storage_type storage_;       		// Q storage type

  Matrix<T, data_block_type> data_;	// Factorized QR matrix
  aligned_array<T> tau_;       		// Additional info on Q
  aligned_array<T> work_;		// workspace for geqrf
};

} // namespace ovxx::lapack

namespace dispatcher
{
template <typename T>
struct Evaluator<op::qrd, be::lapack, T>
{
  static bool const ct_valid = blas::traits<T>::valid;
  typedef lapack::qrd<T, true> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
