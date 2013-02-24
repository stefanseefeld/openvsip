/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_opt_lapack_svd_hpp_
#define vsip_opt_lapack_svd_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/opt/lapack/bindings.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/solver/common.hpp>

namespace vsip
{
namespace impl
{
namespace lapack
{

/// SVD decomposition implementation class.  Common functionality for
/// svd by-value and by-reference classes.
template <typename T>
class Svd
{
  // BLAS/LAPACK require complex data to be in interleaved format.
  typedef Layout<2, col2_type, Stride_unit_dense, Cmplx_inter_fmt> data_LP;
  typedef Strided<2, T, data_LP> data_block_type;

public:
  Svd(length_type, length_type, storage_type, storage_type) 
    VSIP_THROW((std::bad_alloc));
  Svd(Svd const&) VSIP_THROW((std::bad_alloc));

  length_type  rows()     const VSIP_NOTHROW { return m_; }
  length_type  columns()  const VSIP_NOTHROW { return n_; }
  storage_type ustorage() const VSIP_NOTHROW { return ust_; }
  storage_type vstorage() const VSIP_NOTHROW { return vst_; }

  template <typename Block0, typename Block1>
  bool decompose(Matrix<T, Block0>,
		 Vector<scalar_f, Block1>) VSIP_NOTHROW;

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool produ(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  template <mat_op_type       tr,
	    product_side_type ps,
	    typename          Block0,
	    typename          Block1>
  bool prodv(const_Matrix<T, Block0>, Matrix<T, Block1>) VSIP_NOTHROW;

  template <typename Block>
  bool u(index_type, index_type, Matrix<T, Block>) VSIP_NOTHROW;

  template <typename Block>
  bool v(index_type, index_type, Matrix<T, Block>) VSIP_NOTHROW;

  length_type order()  const VSIP_NOTHROW { return p_;}

private:
  typedef typename Scalar_of<T>::type scalar_type;
  typedef std::vector<T, Aligned_allocator<T> > vector_type;
  typedef std::vector<scalar_type, Aligned_allocator<scalar_type> >
		svector_type;

  Svd &operator=(Svd const&) VSIP_NOTHROW;

  length_type  m_;			// Number of rows.
  length_type  n_;			// Number of cols.
  length_type  p_;			// min(rows, cols)
  storage_type ust_;			// U storage type
  storage_type vst_;			// V storage type

  Matrix<T, data_block_type> data_;	// Factorized matrix
  Matrix<T, data_block_type> q_;	// U matrix
  Matrix<T, data_block_type> pt_;	// V' matrix

  vector_type  tauq_;			// Additional info on Q
  vector_type  taup_;			// Additional info on P
  svector_type b_d_;			// Diagonal elements of B
  svector_type b_e_;			// Off-diagonal elements of B
					//  - gebrd requires min(m, n)-1
					//  - bdsqr requires min(m, n)

  length_type lwork_gebrd_;		// size of workspace needed for gebrd
  vector_type work_gebrd_;		// workspace for gebrd
  length_type lwork_gbr_;		// size of workspace needed for gebrd
  vector_type work_gbr_;		// workspace for gebrd
};

length_type inline
select_dim(storage_type st, length_type full, length_type part)
{
  return (st == svd_uvfull) ? full :
         (st == svd_uvpart) ? part : 0;
}

template <typename T>
Svd<T>::Svd(length_type  rows, length_type  cols,
	    storage_type ust, storage_type vst) VSIP_THROW((std::bad_alloc))
  : m_          (rows),
    n_          (cols),
    p_          (std::min(m_, n_)),
    ust_        (ust),
    vst_        (vst),

    data_       (m_, n_),
    q_          (select_dim(ust_, m_, m_), select_dim(ust_, m_, p_)),
    pt_         (select_dim(vst_, n_, p_), select_dim(vst_, n_, n_)),

    tauq_       (p_),
    taup_       (p_),
    b_d_        (p_),
    b_e_        (p_),

    lwork_gebrd_((m_ + n_) * lapack::gebrd_blksize<T>(m_, n_)),
    work_gebrd_ (lwork_gebrd_),
    lwork_gbr_  (p_ * lapack::gbr_blksize<T>('Q', m_, m_, n_)),
    work_gbr_   (lwork_gbr_)
{
  assert(m_ > 0 && n_ > 0);
  assert(ust_ == svd_uvnos || ust_ == svd_uvpart || ust_ == svd_uvfull);
  assert(vst_ == svd_uvnos || vst_ == svd_uvpart || vst_ == svd_uvfull);
}

template <typename T>
Svd<T>::Svd(Svd const& sv) VSIP_THROW((std::bad_alloc))
  : m_          (sv.m_),
    n_          (sv.n_),
    p_          (sv.p_),
    ust_        (sv.ust_),
    vst_        (sv.vst_),

    data_       (m_, n_),
    q_          (select_dim(ust_, m_, m_), select_dim(ust_, m_, p_)),
    pt_         (select_dim(vst_, n_, p_), select_dim(vst_, n_, n_)),

    tauq_       (p_),
    taup_       (p_),
    b_d_        (p_),
    b_e_        (p_),
    lwork_gebrd_((m_ + n_) * lapack::gebrd_blksize<T>(m_, n_)),
    work_gebrd_ (lwork_gebrd_),
    lwork_gbr_  (p_ * lapack::gbr_blksize<T>('Q', m_, m_, n_)),
    work_gbr_   (lwork_gbr_)
{
  data_ = sv.data_;
  q_    = sv.q_;
  pt_   = sv.pt_;
  for (index_type i=0; i<p_; ++i)
  {
    b_d_[i]  = sv.b_d_[i];
    b_e_[i]  = sv.b_e_[i];
    tauq_[i] = sv.tauq_[i];
    taup_[i] = sv.taup_[i];
  }
}

/// Decompose matrix M into
///
/// Return
///   DEST contains M's singular values.
///
/// Requires
///   M to be a full rank, modifiable matrix of ROWS x COLS.
template <typename T>
template <typename Block0, typename Block1>
bool
Svd<T>::decompose(Matrix<T, Block0> m, Vector<vsip::scalar_f, Block1> dest)
  VSIP_NOTHROW
{
  assert(m.size(0) == m_ && m.size(1) == n_);
  assert(dest.size() == p_);

  int lwork   = lwork_gebrd_;

  assign_local(data_, m);

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
    Ext_data<data_block_type> ext(data_.block());

    lapack::gebrd(m_, n_,
		  ext.data(), ext.stride(1),	// A, lda
		  &b_d_[0],			// diagonal of B
		  &b_e_[0],			// off-diagonal of B
		  &tauq_[0],
		  &taup_[0],
		  &work_gebrd_[0], lwork);
    assert((length_type)lwork <= lwork_gebrd_);
    // FLOPS:
    //   scalar : (4/3)*n^2*(3*m-n) for m >= n
    //          : (4/3)*m^2*(3*n-m) for m <  n
    //   complex: 4*
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

    Ext_data<data_block_type> ext_q(q_.block());
    lwork   = lwork_gbr_;
    lapack::gbr('Q', m_, cols, n_,
		ext_q.data(), ext_q.stride(1),	// A, lda
		&tauq_[0],
		&work_gbr_[0], lwork);
    // FLOPS:
    // scalar : To form full Q:
    //        :    (4/3) n (3m^2 - 3mn + n^2) for m >= n
    //        :    (4/3) m^3                  for m < n
    //        : To form n leading columns of Q when m > n:
    //        :    (2/3) n^2 (3m - n^2)
    // complex: 4*
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

    Ext_data<data_block_type> ext_pt(pt_.block());
    lwork   = lwork_gbr_;
    lapack::gbr('P', rows, n_, m_,
		ext_pt.data(), ext_pt.stride(1),	// A, lda
		&taup_[0],
		&work_gbr_[0], lwork);
    // FLOPS:
    // scalar : To form full PT:
    //        :    (4/3) n^3                  for m >= n
    //        :    (4/3) m (3n^2 - 3mn + n^2) for m < n
    //        : To form m leading columns of PT when m < n:
    //        :    (2/3) m^2 (3n - m^2)
    // complex: 4*
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
    Ext_data<data_block_type> ext_q (q_.block());
    Ext_data<data_block_type> ext_pt(pt_.block());

    length_type nru    = (ust_ != svd_uvnos) ? m_ : 0;
    T*          q_ptr  = (ust_ != svd_uvnos) ? ext_q.data()    : 0;
    stride_type q_ld   = (ust_ != svd_uvnos) ? ext_q.stride(1) : 1;

    length_type ncvt   = (vst_ != svd_uvnos) ? n_ : 0;
    T*          pt_ptr = (vst_ != svd_uvnos) ? ext_pt.data()    : 0;
    stride_type pt_ld  = (vst_ != svd_uvnos) ? ext_pt.stride(1) : 1;
    
    // Compute SVD of bidiagonal matrix B.

    // Note: MKL says that work-size need only 4*(p_-1), 
    //       however MKL 5.x needs 4*(p_).
    vector_type work(4*p_);
    char uplo = (m_ >= n_) ? 'U' : 'L';
    lapack::bdsqr(uplo,
		p_,	// Order of matrix B.
		ncvt,	// Number of columns of VT (right singular vectors)
		nru,	// Number of rows of U     (left  singular vectors)
		0,	// Number of columns of C: 0 since no C supplied.
		&b_d_[0],	//
		&b_e_[0],	//
	        pt_ptr, pt_ld,		// [p_ x ncvt]
		q_ptr,  q_ld,		// [nru x p_]
		0, 1,	// Not referenced since ncc = 0
		&work[0]);
    // Flops (scalar):
    //  n^2 (singular values)
    //  6n^2 * nru  (left singular vectors)	(complex 2*)
    //  6n^2 * ncvt (right singular vectors)	(complex 2*)
  }

  for (index_type i=0; i<p_; ++i)
    dest(i) = b_d_[i];

  return true;
}



/// prod_uv() is a set of helper routines for produ() and prodv().

/// It is overloaded on Bool_type<Is_complex<T>::value> to handle
/// transpose and hermetian properly.  (Tranpose is defined for non-complex
/// T, but does not make sense.  Hermetion is only defined for complex
/// T).
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          T,
	  typename          Block0,
	  typename          Block1,
	  typename          Block2>
inline void
prod_uv(const_Matrix<T, Block0> uv,
	const_Matrix<T, Block1> b,
	Matrix<T, Block2>       x,
	Bool_type<false>         /*is_complex*/)
{
  VSIP_IMPL_STATIC_ASSERT(Is_complex<T>::value == false);
  VSIP_IMPL_STATIC_ASSERT(tr != mat_herm);

  if (ps == mat_lside)
  {
    if (tr == mat_ntrans)
    {
      assert(b.size(0) == uv.size(1));
      assert(x.size(0) == uv.size(0));
      assert(b.size(1) == x.size(1));
      x = prod(uv, b);
    }
    else if (tr == mat_trans)
    {
      assert(b.size(0) == uv.size(0));
      assert(x.size(0) == uv.size(1));
      assert(b.size(1) == x.size(1));
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
      assert(b.size(1) == uv.size(0));
      assert(x.size(1) == uv.size(1));
      assert(b.size(0) == x.size(0));
      x = prod(b, uv);
    }
    else if (tr == mat_trans)
    {
      assert(b.size(1) == uv.size(1));
      assert(x.size(1) == uv.size(0));
      assert(b.size(0) == x.size(0));
      x = prod(b, trans(uv));
    }
    else if (tr == mat_herm)
    {
      assert(false);
    }
  }
}

template <mat_op_type       tr,
	  product_side_type ps,
	  typename          T,
	  typename          Block0,
	  typename          Block1,
	  typename          Block2>
inline void
prod_uv(const_Matrix<T, Block0> uv,
	const_Matrix<T, Block1> b,
	Matrix<T, Block2>       x,
	Bool_type<true>         /*is_complex*/)
{
  VSIP_IMPL_STATIC_ASSERT(Is_complex<T>::value == true);
  VSIP_IMPL_STATIC_ASSERT(tr != mat_trans);

  if (ps == mat_lside)
  {
    if (tr == mat_ntrans)
    {
      assert(b.size(0) == uv.size(1));
      assert(x.size(0) == uv.size(0));
      assert(b.size(1) == x.size(1));
      x = prod(uv, b);
    }
    else if (tr == mat_trans)
    {
      assert(0);
    }
    else if (tr == mat_herm)
    {
      assert(b.size(0) == uv.size(0));
      assert(x.size(0) == uv.size(1));
      assert(b.size(1) == x.size(1));
      x = prod(herm(uv), b);
    }
  }
  else /* (ps == mat_rside) */
  {
    if (tr == mat_ntrans)
    {
      assert(b.size(1) == uv.size(0));
      assert(x.size(1) == uv.size(1));
      assert(b.size(0) == x.size(0));
      x = prod(b, uv);
    }
    else if (tr == mat_trans)
    {
      assert(0);
    }
    else if (tr == mat_herm)
    {
      assert(b.size(1) == uv.size(1));
      assert(x.size(1) == uv.size(0));
      assert(b.size(0) == x.size(0));
      x = prod(b, herm(uv));
    }
  }
}



/// Compute product of U and b
///
/// If svd_uvpart: U is (m, p)
/// If svd_uvfull: U is (m, m)
///
/// ustorage   | ps        | tr         | product | b (in) | x (out)
/// svd_uvpart | mat_lside | mat_ntrans | U b     | (p, s) | (m, s)
/// svd_uvpart | mat_lside | mat_trans  | U' b    | (m, s) | (p, s)
/// svd_uvpart | mat_lside | mat_herm   | U* b    | (m, s) | (p, s)
///
/// svd_uvpart | mat_rside | mat_ntrans | b U     | (s, m) | (s, p)
/// svd_uvpart | mat_rside | mat_trans  | b U'    | (s, p) | (s, m)
/// svd_uvpart | mat_rside | mat_herm   | b U*    | (s, p) | (s, m)
///
/// svd_uvfull | mat_lside | mat_ntrans | U b     | (m, s) | (m, s)
/// svd_uvfull | mat_lside | mat_trans  | U' b    | (m, s) | (m, s)
/// svd_uvfull | mat_lside | mat_herm   | U* b    | (m, s) | (m, s)
///
/// svd_uvfull | mat_rside | mat_ntrans | b U     | (s, m) | (s, m)
/// svd_uvfull | mat_rside | mat_trans  | b U'    | (s, m) | (s, m)
/// svd_uvfull | mat_rside | mat_herm   | b U*    | (s, m) | (s, m)

template <typename T>
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          Block0,
	  typename          Block1>
bool
Svd<T>::produ(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  prod_uv<tr, ps>(this->q_, b, x, Bool_type<Is_complex<T>::value>());
  return true;
}



/// Compute product of V and b
///
/// Note: product is with V, not V' (unless asked)
///
/// If svd_uvpart: V is (n, p)
/// If svd_uvfull: V is (n, n)
///
/// ustorage   | ps        | tr         | product | b (in) | x (out)
/// svd_uvpart | mat_lside | mat_ntrans | V b     | (p, s) | (n, s)
/// svd_uvpart | mat_lside | mat_trans  | V' b    | (n, s) | (p, s)
/// svd_uvpart | mat_lside | mat_herm   | V* b    | (n, s) | (p, s)
///
/// svd_uvpart | mat_rside | mat_ntrans | b V     | (s, n) | (s, p)
/// svd_uvpart | mat_rside | mat_trans  | b V'    | (s, p) | (s, n)
/// svd_uvpart | mat_rside | mat_herm   | b V*    | (s, p) | (s, n)
///
/// svd_uvfull | mat_lside | mat_ntrans | V b     | (n, s) | (n, s)
/// svd_uvfull | mat_lside | mat_trans  | V' b    | (n, s) | (n, s)
/// svd_uvfull | mat_lside | mat_herm   | V* b    | (n, s) | (n, s)
///
/// svd_uvfull | mat_rside | mat_ntrans | b V     | (s, n) | (s, n)
/// svd_uvfull | mat_rside | mat_trans  | b V'    | (s, n) | (s, n)
/// svd_uvfull | mat_rside | mat_herm   | b V*    | (s, n) | (s, n)

template <typename T>
template <mat_op_type       tr,
	  product_side_type ps,
	  typename          Block0,
	  typename          Block1>
bool
Svd<T>::prodv(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
{
  prod_uv<tr, ps>(trans_or_herm(this->pt_), b, x,
		  Bool_type<Is_complex<T>::value>());
  return true;
}

/// Return the submatrix U containing columns (low .. high) inclusive.
template <typename T>
template <typename Block>
bool
Svd<T>::u(index_type low, index_type high, Matrix<T, Block> u) VSIP_NOTHROW
{
  assert((ust_ == svd_uvpart && high < p_) || (ust_ == svd_uvfull && high < m_));
  assert(u.size(0) == m_);
  assert(u.size(1) == high - low + 1);

  u = q_(Domain<2>(m_, Domain<1>(low, 1, high-low+1)));

  return true;
}

/// Return the submatrix V containing columns (low .. high) inclusive.
template <typename T>
template <typename Block>
bool
Svd<T>::v(index_type low, index_type high, Matrix<T, Block> v) VSIP_NOTHROW
{
  assert((vst_ == svd_uvpart && high < p_) || (vst_ == svd_uvfull && high < n_));
  assert(v.size(0) == n_);
  assert(v.size(1) == high - low + 1);

  v = trans_or_herm(pt_(Domain<2>(Domain<1>(low, 1, high-low+1), n_)));

  return true;
}

} // namespace vsip::impl::lapack
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T>
struct Evaluator<op::svd, be::lapack, T>
{
  static bool const ct_valid = impl::blas::Blas_traits<T>::valid;
  typedef impl::lapack::Svd<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
