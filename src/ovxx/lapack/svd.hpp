//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_svx_hpp_
#define ovxx_lapack_svx_hpp_

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/impl/math_enum.hpp>
#include <ovxx/lapack/lapack.hpp>
#include <vsip/impl/solver/common.hpp>

namespace ovxx
{
namespace lapack
{
/// prod_uv() is a set of helper routines for produ() and prodv().
/// It is overloaded on integral_constant<bool, is_complex<T>::value> to handle
/// transpose and Hermitian properly.
template <mat_op_type tr, product_side_type ps,
	  typename T, typename B0, typename B1, typename B2>
inline void
prod_uv(const_Matrix<T, B0> uv, const_Matrix<T, B1> b, Matrix<T, B2> x,
	false_type /*is_complex*/)
{
  OVXX_CT_ASSERT(is_complex<T>::value == false);
  OVXX_CT_ASSERT(tr != mat_herm);

  if (ps == mat_lside)
  {
    if (tr == mat_ntrans)
    {
      OVXX_PRECONDITION(b.size(0) == uv.size(1));
      OVXX_PRECONDITION(x.size(0) == uv.size(0));
      OVXX_PRECONDITION(b.size(1) == x.size(1));
      x = prod(uv, b);
    }
    else if (tr == mat_trans)
    {
      OVXX_PRECONDITION(b.size(0) == uv.size(0));
      OVXX_PRECONDITION(x.size(0) == uv.size(1));
      OVXX_PRECONDITION(b.size(1) == x.size(1));
      x = prod(trans(uv), b);
    }
    else if (tr == mat_herm)
    {
      assert(false);
    }
  }
  else /* (ps == mat_rside) */
  {
    if (tr == mat_ntrans)
    {
      OVXX_PRECONDITION(b.size(1) == uv.size(0));
      OVXX_PRECONDITION(x.size(1) == uv.size(1));
      OVXX_PRECONDITION(b.size(0) == x.size(0));
      x = prod(b, uv);
    }
    else if (tr == mat_trans)
    {
      OVXX_PRECONDITION(b.size(1) == uv.size(1));
      OVXX_PRECONDITION(x.size(1) == uv.size(0));
      OVXX_PRECONDITION(b.size(0) == x.size(0));
      x = prod(b, trans(uv));
    }
    else if (tr == mat_herm)
    {
      assert(false);
    }
  }
}

template <mat_op_type tr, product_side_type ps,
	  typename T, typename B0, typename B1, typename B2>
inline void
prod_uv(const_Matrix<T, B0> uv, const_Matrix<T, B1> b, Matrix<T, B2> x,
	true_type /*is_complex*/)
{
  OVXX_CT_ASSERT(is_complex<T>::value == true);
  OVXX_CT_ASSERT(tr != mat_trans);

  if (ps == mat_lside)
  {
    if (tr == mat_ntrans)
    {
      OVXX_PRECONDITION(b.size(0) == uv.size(1));
      OVXX_PRECONDITION(x.size(0) == uv.size(0));
      OVXX_PRECONDITION(b.size(1) == x.size(1));
      x = prod(uv, b);
    }
    else if (tr == mat_trans)
    {
      assert(0);
    }
    else if (tr == mat_herm)
    {
      OVXX_PRECONDITION(b.size(0) == uv.size(0));
      OVXX_PRECONDITION(x.size(0) == uv.size(1));
      OVXX_PRECONDITION(b.size(1) == x.size(1));
      x = prod(herm(uv), b);
    }
  }
  else /* (ps == mat_rside) */
  {
    if (tr == mat_ntrans)
    {
      OVXX_PRECONDITION(b.size(1) == uv.size(0));
      OVXX_PRECONDITION(x.size(1) == uv.size(1));
      OVXX_PRECONDITION(b.size(0) == x.size(0));
      x = prod(b, uv);
    }
    else if (tr == mat_trans)
    {
      assert(0);
    }
    else if (tr == mat_herm)
    {
      OVXX_PRECONDITION(b.size(1) == uv.size(1));
      OVXX_PRECONDITION(x.size(1) == uv.size(0));
      OVXX_PRECONDITION(b.size(0) == x.size(0));
      x = prod(b, herm(uv));
    }
  }
}

template <typename T>
class svd
{
  typedef Layout<2, col2_type, dense, array> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  svd(length_type, length_type, storage_type, storage_type) 
    VSIP_THROW((std::bad_alloc));
  svd(svd const &other) VSIP_THROW((std::bad_alloc));
  svd &operator= (svd const &other) VSIP_THROW((std::bad_alloc));
  length_type rows() const VSIP_NOTHROW { return m_;}
  length_type columns() const VSIP_NOTHROW { return n_;}
  storage_type ustorage() const VSIP_NOTHROW { return ust_;}
  storage_type vstorage() const VSIP_NOTHROW { return vst_;}

  template <typename B0, typename B1>
  bool decompose(Matrix<T, B0>, Vector<float, B1>) VSIP_NOTHROW;

  template <mat_op_type tr, product_side_type ps,
	    typename B0, typename B1>
  bool produ(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    prod_uv<tr, ps>(this->q_, b, x,
		    integral_constant<bool, is_complex<T>::value>());
    return true;
  }

  template <mat_op_type tr, product_side_type ps,
	    typename B0, typename B1>
  bool prodv(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    prod_uv<tr, ps>(impl::trans_or_herm(this->pt_), b, x,
		    integral_constant<bool, is_complex<T>::value>());
    return true;
  }

  template <typename B>
  bool u(index_type low, index_type high, Matrix<T, B> u) VSIP_NOTHROW
  {
    OVXX_PRECONDITION((ust_ == svd_uvpart && high < p_) ||
		      (ust_ == svd_uvfull && high < m_));
    OVXX_PRECONDITION(u.size(0) == m_);
    OVXX_PRECONDITION(u.size(1) == high - low + 1);
    u = q_(Domain<2>(m_, Domain<1>(low, 1, high-low+1)));
    return true;
  }

  template <typename B>
  bool v(index_type low, index_type high, Matrix<T, B> v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION((vst_ == svd_uvpart && high < p_) ||
		      (vst_ == svd_uvfull && high < n_));
    OVXX_PRECONDITION(v.size(0) == n_);
    OVXX_PRECONDITION(v.size(1) == high - low + 1);
    v = impl::trans_or_herm(pt_(Domain<2>(Domain<1>(low, 1, high-low+1), n_)));
    return true;
  }

  length_type order()  const VSIP_NOTHROW { return p_;}

private:
  typedef typename scalar_of<T>::type scalar_type;

  length_type  m_;		    // Number of rows.
  length_type  n_;		    // Number of cols.
  length_type  p_;		    // min(rows, cols)
  storage_type ust_;	            // U storage type
  storage_type vst_;	      	    // V storage type

  Matrix<T, data_block_type> data_; // Factorized matrix
  Matrix<T, data_block_type> q_;    // U matrix
  Matrix<T, data_block_type> pt_;   // V' matrix

  aligned_array<T> tauq_;      	    // Additional info on Q
  aligned_array<T> taup_;      	    // Additional info on P
  aligned_array<scalar_type> b_d_;  // Diagonal elements of B
  aligned_array<scalar_type> b_e_;  // Off-diagonal elements of B
				    //  - gebrd requires min(m, n)-1
				    //  - bdsqr requires min(m, n)

  aligned_array<T> work_;      	    // workspace for gbr & gebrd
};

length_type inline
select_dim(storage_type st, length_type full, length_type part)
{
  return (st == svd_uvfull) ? full :
         (st == svd_uvpart) ? part : 0;
}

template <typename T>
svd<T>::svd(length_type  rows, length_type  cols,
	    storage_type ust, storage_type vst) VSIP_THROW((std::bad_alloc))
  : m_(rows),
    n_(cols),
    p_(std::min(m_, n_)),
    ust_(ust),
    vst_(vst),

    data_(m_, n_),
    q_(select_dim(ust_, m_, m_), select_dim(ust_, m_, p_)),
    pt_(select_dim(vst_, n_, p_), select_dim(vst_, n_, n_)),

    tauq_(p_),
    taup_(p_),
    b_d_(p_),
    b_e_(p_),
    work_(std::max((m_ + n_) * gebrd_work<T>(m_, n_),
		   p_ * lapack::gbr_work<T>('Q', m_, m_, n_)))
{
  OVXX_PRECONDITION(m_ > 0 && n_ > 0);
  OVXX_PRECONDITION(ust_ == svd_uvnos || ust_ == svd_uvpart || ust_ == svd_uvfull);
  OVXX_PRECONDITION(vst_ == svd_uvnos || vst_ == svd_uvpart || vst_ == svd_uvfull);
}

template <typename T>
svd<T>::svd(svd const &other) VSIP_THROW((std::bad_alloc))
  : m_(other.m_),
    n_(other.n_),
    p_(other.p_),
    ust_(other.ust_),
    vst_(other.vst_),

    data_(m_, n_),
    q_(select_dim(ust_, m_, m_), select_dim(ust_, m_, p_)),
    pt_(select_dim(vst_, n_, p_), select_dim(vst_, n_, n_)),

    tauq_(p_),
    taup_(p_),
    b_d_(p_),
    b_e_(p_),
    work_(other.work_.size())
{
  data_ = other.data_;
  q_ = other.q_;
  pt_ = other.pt_;
  for (index_type i = 0; i< p_; ++i)
  {
    b_d_[i]  = other.b_d_[i];
    b_e_[i]  = other.b_e_[i];
    tauq_[i] = other.tauq_[i];
    taup_[i] = other.taup_[i];
  }
}

template <typename T>
template <typename B0, typename B1>
bool
svd<T>::decompose(Matrix<T, B0> m, Vector<float, B1> dest)
  VSIP_NOTHROW
{
  OVXX_PRECONDITION(m.size(0) == m_ && m.size(1) == n_);
  OVXX_PRECONDITION(dest.size() == p_);

  int lwork = work_.size();
  parallel::assign_local(data_, m);

  // Step 1: Reduce general matrix A to bidiagonal form.
  //
  // If m >= n, then
  //   A = Q_1 B_1 P'
  // Where
  //   Q_1 (m, n) orthogonal/unitary
  //   B_1 (n, n) upper diagonal
  //   P'  (n, n) orthogonal/unitary
  //
  // If m < n, then
  //   A = Q_1 B_1 P'
  // Where
  //   Q_1 (m, m) orthogonal/unitary
  //   B_1 (m, m) lower diagonal
  //   P'  (m, n) orthogonal/unitary
  {
    dda::Data<data_block_type, dda::out> data(data_.block());
    lapack::gebrd(m_, n_,
		  data.ptr(), data.stride(1),	// A, lda
		  b_d_.get(),			// diagonal of B
		  b_e_.get(),			// off-diagonal of B
		  tauq_.get(),
		  taup_.get(),
		  work_.get(), lwork);
    OVXX_PRECONDITION((length_type)lwork <= work_.size());
  }

  // Step 2: Generate real orthoganol (complex unitary) matrices Q and P'
  //         determined by gebrd.
  if (ust_ == svd_uvfull || ust_ == svd_uvpart)
  {
    // svd_uvfull: generate whole Q (m_, m_):
    // svd_uvpart: generate first p_ columns of Q (m_, p_):

    length_type cols = (ust_ == svd_uvfull) ? m_ : p_;

    if (m_ >= n_)
      q_(Domain<2>(m_, n_)) = data_;
    else
      q_ = data_(Domain<2>(m_, m_));

    dda::Data<data_block_type, dda::inout> data_q(q_.block());
    lwork   = work_.size();
    lapack::gbr('Q', m_, cols, n_,
		data_q.ptr(), data_q.stride(1),	// A, lda
		tauq_.get(),
		work_.get(), lwork);
  }


  if (vst_ == svd_uvfull || vst_ == svd_uvpart)
  {
    // svd_uvfull: generate whole P' (n_, n_):
    // svd_uvpart: generate first p_ rows of P' (p_, n_):

    length_type rows = (vst_ == svd_uvfull) ? n_ : p_;

    if (m_ >= n_)
      pt_ = data_(Domain<2>(n_, n_));
    else
      pt_(Domain<2>(m_, n_)) = data_;

    dda::Data<data_block_type, dda::inout> data_pt(pt_.block());
    lwork   = work_.size();
    lapack::gbr('P', rows, n_, m_,
		data_pt.ptr(), data_pt.stride(1),	// A, lda
		taup_.get(),
		work_.get(), lwork);
  }


  // Step 3: Form singular value decomposition from the bidiagonal matrix B
  //
  // Factor bidiagonal matrix B into SVD form:
  //   B = Q * S * herm(P)
  //
  // and optionally apply to Q and PT matrices from step 2
  //   A = U * B * VT
  //   A = (U*Q) * S * (herm(P)*VT)
  //
  // After this step:
  //   b_d_ will refer to the singular values,
  //   q_   will refer to the left singular vectors  (U*Q),
  //   pt_  will refer to the right singular vectors (herm(P)*VT)

  {
    dda::Data<data_block_type, dda::inout> data_q(q_.block());
    dda::Data<data_block_type, dda::inout> data_pt(pt_.block());

    length_type nru    = (ust_ != svd_uvnos) ? m_ : 0;
    T*          q_ptr  = (ust_ != svd_uvnos) ? data_q.ptr()    : 0;
    stride_type q_ld   = (ust_ != svd_uvnos) ? data_q.stride(1) : 1;

    length_type ncvt   = (vst_ != svd_uvnos) ? n_ : 0;
    T*          pt_ptr = (vst_ != svd_uvnos) ? data_pt.ptr()    : 0;
    stride_type pt_ld  = (vst_ != svd_uvnos) ? data_pt.stride(1) : 1;
    
    // Compute SVD of bidiagonal matrix B.

    aligned_array<T> work(4*p_);
    char uplo = (m_ >= n_) ? 'U' : 'L';
    lapack::bdsqr(uplo,
		  p_,	// Order of matrix B.
		  ncvt,	// Number of columns of VT (right singular vectors)
		  nru,	// Number of rows of U     (left  singular vectors)
		  0,	// Number of columns of C: 0 since no C supplied.
		  b_d_.get(),	//
		  b_e_.get(),	//
		  pt_ptr, pt_ld,		// [p_ x ncvt]
		  q_ptr,  q_ld,		// [nru x p_]
		  0, 1,	// Not referenced since ncc = 0
		  work.get());
  }

  for (index_type i=0; i<p_; ++i)
    dest.put(i, b_d_[i]);
  return true;
}

} // namespace ovxx::lapack

namespace dispatcher
{
template <typename T>
struct Evaluator<op::svd, be::lapack, T>
{
  static bool const ct_valid = blas::traits<T>::valid;
  typedef lapack::svd<T> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
