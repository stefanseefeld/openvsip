//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_cholesky_hpp_
#define ovxx_lapack_cholesky_hpp_

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <ovxx/lapack/blas.hpp>
#include <ovxx/lapack/lapack.hpp>
#include <vsip/impl/solver/common.hpp>
#include <algorithm>

namespace ovxx
{
namespace lapack
{

template <typename T>
class chold : ct_assert<blas::traits<T>::valid>
{
  typedef Layout<2, col2_type, dense, array> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  chold(mat_uplo uplo, length_type length)
  : uplo_(uplo),
    length_(length),
    data_(length_, length_)
  {
    OVXX_PRECONDITION(length_ > 0);
    OVXX_PRECONDITION(uplo_ == upper || uplo_ == lower);
  }
  chold(chold const &other)
  : uplo_(other.uplo_),
    length_(other.length_),
    data_(length_, length_)
  {
    data_ = other.data_;
  }
  chold &operator=(chold const &other) VSIP_NOTHROW
  {
    // TODO: At present assignment requires dimensions to match,
    //       as views aren't resizable.
    OVXX_PRECONDITION(length_ == other.length_);
    uplo_ = other.uplo_;
    data_ = other.data_;
  }

  length_type length() const { return length_;}
  mat_uplo uplo() const { return uplo_;}

  template <typename B>
  bool decompose(Matrix<T, B> m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(m.size(0) == length_ && m.size(1) == length_);
    data_ = m;
    dda::Data<data_block_type, dda::inout> data(data_.block());
    bool success = lapack::potrf(uplo_ == upper ? 'U' : 'L',
				 length_, data.ptr(), data.stride(1));
    return success;
  }

  template <typename B0, typename B1>
  bool solve(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(b.size(0) == length_);
    OVXX_PRECONDITION(b.size(0) == x.size(0) && b.size(1) == x.size(1));
    Matrix<T, data_block_type> b_clone(b.size(0), b.size(1));
    b_clone = b;
    {
      dda::Data<data_block_type, dda::inout> b_data(b_clone.block());
      dda::Data<data_block_type, dda::in> a_data(data_.block());
      lapack::potrs(uplo_ == upper ? 'U' : 'L',
		    length_,
		    b.size(1),
		    a_data.ptr(), a_data.stride(1),
		    b_data.ptr(), b_data.stride(1));
    }
    x = b_clone;
    return true;
  }

private:
  mat_uplo uplo_;
  length_type length_;
  Matrix<T, data_block_type> data_;
};

} // namespace ovxx::lapack

namespace dispatcher
{
template <typename T>
struct Evaluator<op::chold, be::lapack, T>
{
  // The Lapack Cholesky solver supports all BLAS types.
  static bool const ct_valid = blas::traits<T>::valid;
  typedef lapack::chold<T> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
