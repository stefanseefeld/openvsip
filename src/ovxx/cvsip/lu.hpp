//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_lu_hpp_
#define ovxx_cvsip_lu_hpp_

#include <algorithm>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/math_enum.hpp>
#include <vsip/impl/solver/common.hpp>
#include <ovxx/cvsip/solver.hpp>
#include <ovxx/cvsip/view.hpp>

namespace ovxx
{
namespace cvsip
{
template <typename T>
class lu_solver : ct_assert<solver_traits<T>::valid>
{
  typedef solver_traits<T> traits;
  typedef Layout<2, row2_type, dense, interleaved_complex> data_layout_type;
  typedef Strided<2, T, data_layout_type> data_block_type;

public:
  lu_solver(length_type length)
    : length_(length),
      data_(length_, length_),
      cvsip_data_(data_.block().ptr(), length_, length_, true),
      lu_(traits::lu_create(length_))
  { OVXX_PRECONDITION(length_ > 0);}
  lu_solver(lu_solver const &other)
    : length_(other.length_),
      data_(length_, length_),
      cvsip_data_(data_.block().ptr(), length_, length_, true),
      lu_(traits::lu_create(length_))
  { data_ = other.data_;}
  ~lu_solver() VSIP_NOTHROW { traits::lu_destroy(lu_);}

  length_type length()const VSIP_NOTHROW { return length_;}

  template <typename Block>
  bool decompose(Matrix<T, Block> m) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(m.size(0) == length_ && m.size(1) == length_);

    cvsip_data_.block().release(false);
    assign_local(data_, m);
    cvsip_data_.block().admit(true);
    bool success = !traits::lu_decompose(lu_, cvsip_data_.ptr());
    return success;
  }

  template <mat_op_type tr, typename Block0, typename Block1>
  bool solve(const_Matrix<T, Block0> b, Matrix<T, Block1> x)
  {
    typedef typename get_block_layout<Block0>::order_type order_type;
    static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
    typedef Layout<2, order_type, dense, storage_format> data_LP;
    typedef Strided<2, T, data_LP, Local_map> block_type;

    OVXX_PRECONDITION(b.size(0) == length_);
    OVXX_PRECONDITION(b.size(0) == x.size(0) && b.size(1) == x.size(1));

    Matrix<T, block_type> b_int(b.size(0), b.size(1));
    assign_local(b_int, b);

    if (tr == mat_conj || 
        (tr == mat_trans && is_complex<T>::value) ||
        (tr == mat_herm && !is_complex<T>::value))
      VSIP_IMPL_THROW(unimplemented(
        "LU solver (CVSIP backend) does not implement this transformation"));
    {
      dda::Data<block_type, dda::inout> b_data(b_int.block());

      cvsip::View<2,T,true>
        cvsip_b_int(b_data.ptr(), 0, b_data.stride(0), b_data.size(0),
                    b_data.stride(1), b_data.size(1));

      cvsip_b_int.block().admit(true);
      traits::lu_solve(lu_, tr, cvsip_b_int.ptr());
      cvsip_b_int.block().release(true);
    }
    assign_local(x, b_int);
    return true;
  }

private:
  lu_solver &operator=(lu_solver const&) VSIP_NOTHROW;

  length_type  length_;
  Matrix<T, data_block_type> data_;
  cvsip::View<2,T,true>      cvsip_data_;
  typename traits::lud_type *lu_;
};

} // namespace ovxx::cvsip

namespace dispatcher
{
template <typename T>
struct Evaluator<op::lud, be::cvsip, T>
{
  // The CVSIP LU solver supports all CVSIP types.
  static bool const ct_valid = cvsip::solver_traits<T>::valid;
  typedef cvsip::lu_solver<T> backend_type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
