//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

/// Description:
///   VSIPL++ Library: Cholesky linear system solver using cvsip.

#ifndef VSIP_CORE_CVSIP_CHOLESKY_HPP
#define VSIP_CORE_CVSIP_CHOLESKY_HPP

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

namespace vsip
{
namespace impl
{
namespace cvsip
{
/// Cholesky factorization implementation class.  Common functionality
/// for chold by-value and by-reference classes.
template <typename T>
class Chold : Compile_time_assert<Solver_traits<T>::valid>
{
  typedef Solver_traits<T> traits;
  static storage_format_type const storage_format = dense_complex_format;
  typedef Layout<2, row2_type, dense, storage_format> data_LP;
  typedef Strided<2, T, data_LP> data_block_type;

public:
  Chold(mat_uplo uplo, length_type length)
    : uplo_(uplo),
      length_(length),
      data_(length_, length_),
      cvsip_data_(data_.block().ptr(), length_, length_, true),
      chol_(traits::chol_create(length_, uplo_))
  { assert(length_ > 0);}

  Chold(Chold const &chol)
    : uplo_(chol.uplo_),
    length_(chol.length_),
    data_(length_, length_),
    cvsip_data_(data_.block().ptr(), length_, length_, true),
    chol_(traits::chol_create(length_, uplo_))
  { data_ = chol.data_;}

  ~Chold() { traits::chol_destroy(chol_);}

  mat_uplo    uplo()  const VSIP_NOTHROW { return uplo_;}
  length_type length()const VSIP_NOTHROW { return length_;}

  /// Form Cholesky factorization of matrix A
  ///
  /// Requires
  ///   A to be a square matrix, either
  ///     symmetric positive definite (T real), or
  ///     hermitian positive definite (T complex).
  ///
  /// FLOPS:
  ///   real   : (1/3) n^3
  ///   complex: (4/3) n^3
  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    assert(m.size(0) == length_ && m.size(1) == length_);

    cvsip_data_.block().release(false);
    assign_local(data_, m);
    cvsip_data_.block().admit(true);
    return !traits::chol_decompose(chol_, cvsip_data_.ptr());
  }

  /// Solve A x = b (where A previously given to decompose)
  template <typename Block0, typename Block1>
  bool solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x) VSIP_NOTHROW
  {
    typedef typename get_block_layout<Block0>::order_type order_type;
    static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
    typedef Layout<2, order_type, dense, storage_format> data_LP;
    typedef Strided<2, T, data_LP, Local_map> block_type;

    assert(b.size(0) == length_);
    assert(b.size(0) == x.size(0) && b.size(1) == x.size(1));

    Matrix<T, block_type> b_int(b.size(0), b.size(1));
    assign_local(b_int, b);
    {
      dda::Data<block_type, dda::inout> b_data(b_int.block());
      cvsip::View<2,T,true>
        cvsip_b_int(b_data.ptr(), 0, b_data.stride(0), b_data.size(0),
                    b_data.stride(1), b_data.size(1));

      cvsip_b_int.block().admit(true);
      traits::chol_solve(chol_, cvsip_b_int.ptr());
      cvsip_b_int.block().release(true);
    }
    assign_local(x, b_int);
    return true;
  }

private:
  typedef std::vector<float, Aligned_allocator<float> > vector_type;

  Chold &operator=(Chold const&) VSIP_NOTHROW;

  mat_uplo     uplo_;			// A upper/lower triangular
  length_type  length_;			// Order of A.

  Matrix<T, data_block_type> data_;	// Factorized Cholesky matrix (A)
  View<2,T,true>      cvsip_data_;
  typename traits::chol_type *chol_;
};

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T>
struct Evaluator<op::chold, be::cvsip, T>
{
  static bool const ct_valid = impl::cvsip::Solver_traits<T>::valid;
  typedef impl::cvsip::Chold<T> backend_type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
