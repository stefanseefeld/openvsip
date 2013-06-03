//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_lapack_lu_hpp_
#define ovxx_lapack_lu_hpp_

#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <ovxx/lapack/lapack.hpp>
#include <vsip/impl/solver/common.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/aligned_array.hpp>

namespace ovxx
{
namespace lapack
{

template <typename T>
class lud
{
  typedef Layout<2, col2_type, dense, array> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  lud(length_type length) VSIP_THROW((std::bad_alloc))
  : length_(length),
    ipiv_(length_),
    data_(length_, length_)
  {
    OVXX_PRECONDITION(length_ > 0);
  }
  lud(lud const &other) VSIP_THROW((std::bad_alloc))
  : length_(other.length_),
    ipiv_(length_),
    data_(length_, length_)
  {
    data_ = other.data_;
    for (index_type i = 0; i < length_; ++i)
      ipiv_[i] = other.ipiv_[i];
  }
  lud &operator=(lud const &other) VSIP_NOTHROW
  {
    // TODO: At present assignment requires dimensions to match,
    //       as views aren't resizable.
    OVXX_PRECONDITION(length_ == other.length_);
    ipiv_ = other.ipiv_;
    data_ = other.data_;
  }

  length_type length()const VSIP_NOTHROW { return length_;}

  template <typename B>
  bool decompose(Matrix<T, B> m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(m.size(0) == length_ && m.size(1) == length_);
    parallel::assign_local(data_, m);
    dda::Data<data_block_type, dda::inout> data(data_.block());
    bool success = lapack::getrf(length_, length_,
				 data.ptr(), data.stride(1),	// matrix A, ldA
				 ipiv_.get());			// pivots
    return success;
  }

  template <mat_op_type tr, typename B0, typename B1>
  bool solve(const_Matrix<T, B0> b, Matrix<T, B1> x) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(b.size(0) == length_);
    OVXX_PRECONDITION(b.size(0) == x.size(0) && b.size(1) == x.size(1));
    char trans;
    if (tr == mat_ntrans) trans = 'N';
    else if (tr == mat_trans) trans = 'T';
    else if (tr == mat_herm)
    {
      OVXX_PRECONDITION(is_complex<T>::value);
      trans = 'C';
    }
    Matrix<T, data_block_type> b_int(b.size(0), b.size(1));
    parallel::assign_local(b_int, b);
    {
      dda::Data<data_block_type, dda::inout> b_data(b_int.block());
      dda::Data<data_block_type, dda::in> a_data(data_.block());
      
      getrs(trans,
	    length_,			     // order of A
	    b.size(1),			     // nrhs: number of RH sides
	    a_data.ptr(), a_data.stride(1),  // A, lda
	    ipiv_.get(),	       	     // pivots
	    b_data.ptr(), b_data.stride(1)); // B, ldb
    }
    parallel::assign_local(x, b_int);
    return true;
  }

private:
  length_type length_;
  aligned_array<int> ipiv_;
  Matrix<T, data_block_type> data_;
};

} // namespace ovxx::lapack

namespace dispatcher
{
template <typename T>
struct Evaluator<op::lud, be::lapack, T>
{
  static bool const ct_valid = blas::traits<T>::valid;
  typedef lapack::lud<T> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
