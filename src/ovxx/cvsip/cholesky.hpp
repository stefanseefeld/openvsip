//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_cholesky_hpp_
#define ovxx_cvsip_cholesky_hpp_

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
class chold : ct_assert<solver_traits<T>::valid>
{
  typedef solver_traits<T> traits;
  static storage_format_type const storage_format = vsip::impl::dense_complex_format;
  typedef Layout<2, row2_type, dense, storage_format> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  chold(mat_uplo uplo, length_type length)
    : uplo_(uplo),
      length_(length),
      data_(length_, length_),
      cvsip_data_(data_.block().ptr(), length_, length_, true),
      chol_(traits::chol_create(length_, uplo_))
  { OVXX_PRECONDITION(length_ > 0);}

  chold(chold const &other)
    : uplo_(other.uplo_),
    length_(other.length_),
    data_(length_, length_),
    cvsip_data_(data_.block().ptr(), length_, length_, true),
    chol_(traits::chol_create(length_, uplo_))
  { data_ = other.data_;}

  ~chold() { traits::chol_destroy(chol_);}

  mat_uplo    uplo()  const VSIP_NOTHROW { return uplo_;}
  length_type length()const VSIP_NOTHROW { return length_;}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(m.size(0) == length_ && m.size(1) == length_);

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

    OVXX_PRECONDITION(b.size(0) == length_);
    OVXX_PRECONDITION(b.size(0) == x.size(0) && b.size(1) == x.size(1));

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
  chold &operator=(chold const&) VSIP_NOTHROW;

  mat_uplo     uplo_;			// A upper/lower triangular
  length_type  length_;			// Order of A.

  Matrix<T, data_block_type> data_;	// Factorized Cholesky matrix (A)
  View<2,T,true>      cvsip_data_;
  typename traits::chol_type *chol_;
};

} // namespace ovxx::cvsip

namespace dispatcher
{
template <typename T>
struct Evaluator<op::chold, be::cvsip, T>
{
  static bool const ct_valid = cvsip::solver_traits<T>::valid;
  typedef cvsip::chold<T> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
