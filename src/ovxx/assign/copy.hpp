//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_assign_copy_hpp_
#define ovxx_assign_copy_hpp_

#include <ovxx/assign_fwd.hpp>
#include <ovxx/is_same_ptr.hpp>
#include <vsip/dda.hpp>
#include <cstring>

namespace ovxx
{
namespace assignment
{
// in-place transpose
template <typename T>
void
transpose(T *data,
	  stride_type row_stride,
	  stride_type col_stride,
	  length_type rows,
	  length_type cols)
{
  for (index_type i=0; i<rows; ++i)
    for (index_type j=i; j<cols; ++j)
    {
      std::swap(data[col_stride * i + row_stride * j],
		data[col_stride * j + row_stride * i]);
    }
}

// in-place transpose
template <typename T>
void
transpose(std::pair<T*, T*> const &d,
	  stride_type row_stride,
	  stride_type col_stride,
	  length_type rows,
	  length_type cols)
{
  transpose(d.first,  row_stride, col_stride, rows, cols);
  transpose(d.second, row_stride, col_stride, rows, cols);
}

template <typename T1, typename T2>
void
transpose(T1 *lhs, stride_type lhs_col_stride,
	  T2 const *rhs, stride_type rhs_row_stride,
	  length_type lhs_rows, length_type lhs_cols)
{
  for (index_type r = 0; r != lhs_rows; ++r)
    for (index_type c = 0; c != lhs_cols; ++c)
      lhs[r+c*lhs_col_stride] = rhs[r*rhs_row_stride+c];
}

template <typename T>
void
copy(T *lhs, stride_type lhs_stride,
     T const *rhs, stride_type rhs_stride,
     length_type rows, length_type cols)
{
  // dense
  if (lhs_stride == cols && rhs_stride == cols)
    std::memcpy(lhs, rhs, rows*cols*sizeof(T));
  // unit-stride (contiguous rows)
  else
    for (index_type i = 0; i < rows;
	 ++i, lhs += lhs_stride, rhs += rhs_stride)
      std::memcpy(lhs, rhs, cols*sizeof(T));
}

template <typename T>
void
copy(std::pair<T*, T*> const &lhs, stride_type lhs_stride,
     std::pair<T const*, T const*> const &rhs, stride_type rhs_stride,
     length_type rows, length_type cols)
{
  copy(lhs.first, lhs_stride, rhs.first, rhs_stride, rows, cols);
  copy(lhs.second, lhs_stride, rhs.second, rhs_stride, rows, cols);
}

template <typename T>
void
copy(T *lhs, stride_type lhs_row_stride, stride_type lhs_col_stride,
     T const *rhs, stride_type rhs_row_stride, stride_type rhs_col_stride,
     length_type rows, length_type cols)
{
  if (lhs_col_stride == 1 && rhs_col_stride == 1)
  {
    // dense
    if (lhs_row_stride == static_cast<stride_type>(cols) &&
	rhs_row_stride == static_cast<stride_type>(cols))
      std::memcpy(lhs, rhs, rows*cols*sizeof(T));
    // unit-stride (contiguous rows)
    else
      for (index_type i = 0; i < rows; ++i)
      {
	std::memcpy(lhs, rhs, cols*sizeof(T));
	lhs += lhs_row_stride;
	rhs += rhs_row_stride;
      }
  }
  else
  {
    for (index_type i = 0; i < rows;
	 ++i, lhs += lhs_row_stride, rhs += rhs_row_stride)
    {
      T *lhs_row = lhs;
      T const *rhs_row = rhs;      
      for (index_type j = 0; j != cols;
	   ++j, lhs_row += lhs_col_stride, rhs_row += rhs_col_stride)
	*lhs_row = *rhs_row;
    }
  }
}

template <typename T>
void
copy(std::pair<T*, T*> const &lhs,
     stride_type lhs_row_stride, stride_type lhs_col_stride,
     std::pair<T const*, T const*> const &rhs,
     stride_type rhs_row_stride, stride_type rhs_col_stride,
     length_type rows, length_type cols)
{
  copy(lhs.first, lhs_row_stride, lhs_col_stride,
       rhs.first, rhs_row_stride, rhs_col_stride,
       rows, cols);
  copy(lhs.second, lhs_row_stride, lhs_col_stride,
       rhs.second, rhs_row_stride, rhs_col_stride,
       rows, cols);
}

template <typename LHS, typename RHS>
void copy(LHS &lhs, RHS const &rhs, row2_type, row2_type)
{
  vsip::dda::Data<LHS, dda::out> lhs_data(lhs);
  vsip::dda::Data<RHS, dda::in> rhs_data(rhs);

  copy(lhs_data.ptr(), lhs_data.stride(0), lhs_data.stride(1),
       rhs_data.ptr(), rhs_data.stride(0), rhs_data.stride(1),
       lhs_data.size(0), lhs_data.size(1));
}

template <typename LHS, typename RHS>
void copy(LHS &lhs, RHS const &rhs, col2_type, col2_type)
{
  vsip::dda::Data<LHS, dda::out> lhs_data(lhs);
  vsip::dda::Data<RHS, dda::in> rhs_data(rhs);

  copy(lhs_data.ptr(), lhs_data.stride(1), lhs_data.stride(0),
       rhs_data.ptr(), rhs_data.stride(1), rhs_data.stride(0),
       lhs_data.size(1), lhs_data.size(0));
}

template <typename LHS, typename RHS>
void copy(LHS &lhs, RHS const &rhs, col2_type, row2_type)
{
  vsip::dda::Data<LHS, dda::out> lhs_data(lhs);
  vsip::dda::Data<RHS, dda::in> rhs_data(rhs);

  if (is_same_ptr(lhs_data.ptr(), rhs_data.ptr()))
  {
    // in-place transpose
    OVXX_PRECONDITION(lhs.size(2, 0) == lhs.size(2, 1));
    transpose(lhs_data.ptr(),
	      lhs_data.stride(0), lhs_data.stride(1),
	      lhs.size(2, 0), lhs.size(2, 1));
  }
  else if (lhs_data.stride(0) == 1 && rhs_data.stride(1) == 1)
  {
    transpose(lhs_data.ptr(), lhs_data.stride(1),
	      rhs_data.ptr(), rhs_data.stride(0),
	      lhs.size(2, 0), lhs.size(2, 1));
  }
  else
  {
    copy(lhs_data.ptr(), lhs_data.stride(0), lhs_data.stride(1),

	 rhs_data.ptr(), rhs_data.stride(0), rhs_data.stride(1),
	 lhs.size(2, 0), lhs.size(2, 1));
  }
}

template <typename LHS, typename RHS>
void copy(LHS &lhs, RHS const &rhs, row2_type, col2_type)
{
  vsip::dda::Data<LHS, dda::out> lhs_data(lhs);
  vsip::dda::Data<RHS, dda::in> rhs_data(rhs);

  if (is_same_ptr(lhs_data.ptr(), rhs_data.ptr()))
  {
    // in-place transpose
    OVXX_PRECONDITION(lhs.size(2, 0) == lhs.size(2, 1));
    transpose(lhs_data.ptr(),
	      lhs_data.stride(0), lhs_data.stride(1),
	      lhs.size(2, 0), lhs.size(2, 1));
  }
  else if (lhs_data.stride(1) == 1 && rhs_data.stride(0) == 1)
  {
    transpose(lhs_data.ptr(), lhs_data.stride(0),
	      rhs_data.ptr(), rhs_data.stride(1),
	      lhs.size(2, 1), lhs.size(2, 0));
  }
  else
  {
    copy(lhs_data.ptr(), lhs_data.stride(0), lhs_data.stride(1),
	 rhs_data.ptr(), rhs_data.stride(0), rhs_data.stride(1),
	 lhs.size(2, 1), lhs.size(2, 0));
  }
}

} // namespace ovxx::assignment

namespace dispatcher
{

template <typename LHS, typename RHS>
struct Evaluator<op::assign<1>, be::copy, void(LHS &, RHS const &)>
{
  typedef typename 
  adjust_layout_dim<1, 
    typename adjust_layout_storage_format<array, 
      typename get_block_layout<LHS>::type>::type>::type
    lhs_layout_type;

  typedef typename 
  adjust_layout_dim<1,
    typename adjust_layout_storage_format<array, 
      typename get_block_layout<RHS>::type>::type>::type
  rhs_layout_type;

  static bool const ct_valid = 
    LHS::dim == 1 && RHS::dim == 1 &&
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<RHS, dda::in>::ct_cost == 0 &&
    !is_split_block<LHS>::value &&
    !is_split_block<RHS>::value;

  static std::string name() { return OVXX_DISPATCH_EVAL_NAME;}
  static bool rt_valid(LHS &, RHS const &) { return true;}  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    dda::Data<LHS, dda::out, lhs_layout_type> lhs_data(lhs);
    dda::Data<RHS, dda::in, rhs_layout_type> rhs_data(rhs);

    typename dda::Data<LHS, dda::out, lhs_layout_type>::ptr_type ptr1 = lhs_data.ptr();
    typename dda::Data<RHS, dda::in, rhs_layout_type>::ptr_type ptr2 = rhs_data.ptr();

    stride_type stride1 = lhs_data.stride(0);
    stride_type stride2 = rhs_data.stride(0);
    length_type size    = lhs_data.size(0);
    OVXX_PRECONDITION(size <= rhs_data.size(0));

    if (is_same<typename LHS::value_type, typename RHS::value_type>::value &&
	stride1 == 1 && stride2 == 1)
    {
      std::memcpy(ptr1, ptr2, size*sizeof(typename LHS::value_type));
    }
    else
    {
      while (size--)
      {
	*ptr1 = *ptr2;
	ptr1 += stride1;
	ptr2 += stride2;
      }
    }
  }
};

/// 2D copy assignment. This includes transpose depending
/// on the blocks' dimension-ordering.
template <typename LHS, typename RHS>
struct Evaluator<op::assign<2>, be::copy, void(LHS &, RHS const &)>
{
  static std::string name() { return OVXX_DISPATCH_EVAL_NAME;}

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static bool const is_rhs_expr   = is_expr_block<RHS>::value;

  static bool const is_rhs_simple =
    is_simple_distributed_block<RHS>::value;

  static bool const is_lhs_split  = is_split_block<LHS>::value;
  static bool const is_rhs_split  = is_split_block<RHS>::value;

  static int const  lhs_cost      = dda::Data<LHS, dda::out>::ct_cost;
  static int const  rhs_cost      = dda::Data<RHS, dda::in>::ct_cost;

  typedef typename get_block_layout<RHS>::order_type rhs_order_type;
  typedef typename get_block_layout<LHS>::order_type lhs_order_type;

  static bool const ct_valid =
    is_same<rhs_value_type, lhs_value_type>::value &&
    !is_rhs_expr &&
    lhs_cost == 0 && rhs_cost == 0 &&
    (is_lhs_split == is_rhs_split);

  static bool rt_valid(LHS &, RHS const &) { return true;}

  static void exec(LHS &lhs, RHS const &rhs)
  {
    assignment::copy(lhs, rhs, lhs_order_type(), rhs_order_type());
  }
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
