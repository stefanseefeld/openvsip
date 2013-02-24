/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_opt_lapack_lu_hpp_
#define vsip_opt_lapack_lu_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/math_enum.hpp>
#include <vsip/opt/lapack/bindings.hpp>
#include <vsip/core/temp_buffer.hpp>
#include <vsip/core/solver/common.hpp>
#include <vsip/opt/dispatch.hpp>

namespace vsip
{
namespace impl
{
namespace lapack
{

/// LU factorization implementation class.  Common functionality
/// for lud by-value and by-reference classes.
template <typename T>
class Lu_solver
{
  // BLAS/LAPACK require complex data to be in interleaved format.
  typedef Layout<2, col2_type, Stride_unit_dense, Cmplx_inter_fmt> data_LP;
  typedef Strided<2, T, data_LP> data_block_type;

  // Constructors, copies, assignments, and destructors.
public:
  Lu_solver(length_type length) VSIP_THROW((std::bad_alloc))
  : length_ (length),
    ipiv_   (length_),
    data_   (length_, length_)
  {
    assert(length_ > 0);
  }
  Lu_solver(Lu_solver const &lu) VSIP_THROW((std::bad_alloc))
  : length_ (lu.length_),
    ipiv_   (length_),
    data_   (length_, length_)
  {
    data_ = lu.data_;
    for (index_type i = 0; i < length_; ++i)
      ipiv_[i] = lu.ipiv_[i];
  }

  length_type length()const VSIP_NOTHROW { return length_; }

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    assert(m.size(0) == length_ && m.size(1) == length_);
    assign_local(data_, m);
    Ext_data<data_block_type> ext(data_.block());
    bool success = lapack::getrf(length_, length_,
				 ext.data(), ext.stride(1),	// matrix A, ldA
				 &ipiv_[0]);			// pivots
    return success;
  }

  /// Solve Op(A) x = b (where A previously given to decompose)
  ///
  /// Op(A) is
  ///   A   if tr == mat_ntrans
  ///   A^T if tr == mat_trans
  ///   A'  if tr == mat_herm (valid for T complex only)
  ///
  /// Requires
  ///   B to be a (length, P) matrix
  ///   X to be a (length, P) matrix
  ///
  /// Effects:
  ///   X contains solution to Op(A) X = B
  template <mat_op_type tr, typename Block0, typename Block1>
  bool solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  {
    assert(b.size(0) == length_);
    assert(b.size(0) == x.size(0) && b.size(1) == x.size(1));
    char trans;
    if (tr == mat_ntrans) trans = 'N';
    else if (tr == mat_trans) trans = 'T';
    else if (tr == mat_herm)
    {
      assert(Is_complex<T>::value);
      trans = 'C';
    }
    Matrix<T, data_block_type> b_int(b.size(0), b.size(1));
    assign_local(b_int, b);
    {
      Ext_data<data_block_type> b_ext(b_int.block());
      Ext_data<data_block_type> a_ext(data_.block());
      
      getrs(trans,
	    length_,			  // order of A
	    b.size(1),			  // nrhs: number of RH sides
	    a_ext.data(), a_ext.stride(1),  // A, lda
	    &ipiv_[0],			  // pivots
	    b_ext.data(), b_ext.stride(1)); // B, ldb
    }
    assign_local(x, b_int);
    return true;
  }

private:
  typedef std::vector<int, Aligned_allocator<int> > vector_type;
  Lu_solver &operator=(Lu_solver const&) VSIP_NOTHROW;

  length_type  length_;			// Order of A.
  vector_type  ipiv_;			// Additional info on Q

  Matrix<T, data_block_type> data_;	// Factorized Cholesky matrix (A)
};

} // namespace vsip::impl::lapack
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T>
struct Evaluator<op::lud, be::lapack, T>
{
  // The Lapack LU solver supports all BLAS types.
  static bool const ct_valid = impl::blas::Blas_traits<T>::valid;
  typedef impl::lapack::Lu_solver<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
