/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/extdata.cpp
    @author  Jules Bergmann
    @date    2005-02-11
    @brief   VSIPL++ Library: Unit tests for DDI.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;

/***********************************************************************
  Definitions
***********************************************************************/

// Sum values in a dense block via Direct_data low-level interface.

float
raw_sum(Dense<1, float>& block)
{
  vsip::dda::impl::Accessor<
    Dense<1, float>,
    Dense<1, float>::layout_type,
      vsip::dda::impl::Direct_access_tag>
    raw(block);
  raw.begin(&block, true);
  float*      data   = raw.ptr(&block);
  stride_type stride = raw.stride(&block, 0);

  float sum = 0.0f;
  
  for (index_type i=0; i<block.size(); ++i)
  {
    sum += *data;
    data += stride;
  }
  raw.end(&block, false);

  return sum;
}



// Sum values in a dense block via dda::Data interface.

float
ext_sum(Dense<1, float>& block)
{
  vsip::dda::Data<Dense<1, float>, vsip::dda::in> raw(block);
  float const *data   = raw.ptr();
  stride_type stride = raw.stride(0);

  float sum = 0.0f;
  
  for (index_type i=0; i<block.size(); ++i)
  {
    sum += *data;
    data += stride;
  }

  return sum;
}



// Sum values in a dense block via block interface.
float
blk_sum(Dense<1, float>& block)
{
  float sum = 0.0f;
  
  for (index_type i=0; i<block.size(); ++i)
    sum += block.get(i);

  return sum;
}



// Sum values in a block via dda::Data interface.

/// For Dense, this should use data_interface::Direct_data interface
/// For Strided, this should use data_interface::Copy_data interface

template <typename Block>
typename Block::value_type
gen_ext_sum(Block& block)
{
  typedef typename Block::value_type value_type;
  
  vsip::dda::Data<Block, vsip::dda::in> raw(block);

  value_type const *data   = raw.ptr();
  stride_type stride = raw.stride(0);

  value_type sum = value_type();
  
  for (index_type i=0; i<block.size(); ++i)
  {
    sum += *data;
    data += stride;
  }

  return sum;
}



// Sum values in a block via block interface.

template <typename Block>
typename Block::value_type
gen_blk_sum(Block const& block)
{
  typedef typename Block::value_type value_type;

  value_type sum = value_type();
  
  for (index_type i=0; i<block.size(); ++i)
    sum += block.get(i);

  return sum;
}



// Sum values in a view using view-interface.

template <typename T,
	  typename Block>
T
sum_view(const_Vector<T, Block> view)
{
  T sum = T();
  for (index_type i=0; i<view.size(); ++i)
    sum += view.get(i);
  return sum;
}



// Sum values in a view using dda::Data-interface.

template <typename T,
	  typename Block>
T
sum_ext(const_Vector<T, Block> view)
{
  vsip::dda::Data<Block, dda::in> raw(view.block());
  float const *data   = raw.ptr();
  stride_type stride = raw.stride(0);

  T sum = T();
  
  for (index_type i=0; i<view.size(); ++i)
  {
    sum  += *data;
    data += stride;
  }

  return sum;
}



/// Use external-data interface for element-wise vector addition.

template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
vector_add(
  const_Vector<T, Block1> res,
  const_Vector<T, Block2> op1,
  const_Vector<T, Block3> op2)
{
  vsip::dda::Data<Block1, vsip::dda::out> raw_res(res.block());
  float*   p_raw   = raw_res.ptr();
  stride_type str_raw = raw_res.stride(0);

  vsip::dda::Data<Block2, vsip::dda::in> raw1(op1.block());
  float const *p1   = raw1.ptr();
  stride_type str1 = raw1.stride(0);

  vsip::dda::Data<Block3, vsip::dda::in> raw2(op2.block());
  float const *p2   = raw2.ptr();
  stride_type str2 = raw2.stride(0);

  for (index_type i=0; i<res.size(); ++i)
  {
    *p_raw = *p1 + *p2;

    p1    += str1;
    p2    += str2;
    p_raw += str_raw;
  }
}



// Dot-product of two views using view interface.

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
typename Promotion<T1, T2>::type
dotp_view(
  const_Vector<T1, Block1> op1,
  const_Vector<T2, Block2> op2)
{
  typedef typename Promotion<T1, T2>::type value_type;

  test_assert(op1.size() == op2.size());

  value_type sum = value_type();
  
  for (index_type i=0; i<op1.size(); ++i)
  {
    sum  += op1.get(i) * op2.get(i);
  }

  return sum;
}



// Dot-product of two views using dda::Data interface.

template <typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
typename Promotion<T1, T2>::type
dotp_ext(
  const_Vector<T1, Block1> op1,
  const_Vector<T2, Block2> op2)
{
  typedef typename Promotion<T1, T2>::type value_type;

  test_assert(op1.size() == op2.size());

  vsip::dda::Data<Block1, dda::in> raw1(op1.block());
  T1 const *p1   = raw1.ptr();
  stride_type str1 = raw1.stride(0);

  vsip::dda::Data<Block2, dda::in> raw2(op2.block());
  T2 const *p2   = raw2.ptr();
  stride_type str2 = raw2.stride(0);

  value_type sum = value_type();
  
  for (index_type i=0; i<op1.size(); ++i)
  {
    sum  += *p1 * *p2;
    p1 += str1;
    p2 += str2;
  }

  return sum;
}



// Test various block summations.

void
test_block_sum()
{
  length_type const size = 10;

  Dense<1, float> block(size,   0.f);
  impl::Strided<1, float> pb(size, 0.f);

  block.put(0, 1.f);
  block.put(1, 3.14f);
  block.put(2, 2.78f);

  pb.put(0, 1.f);
  pb.put(1, 3.14f);
  pb.put(2, 2.78f);

  float sum = 1.f + 3.14f + 2.78f;

  test_assert(equal(sum, raw_sum(block)));
  test_assert(equal(sum, ext_sum(block)));
  test_assert(equal(sum, blk_sum(block)));

  test_assert(equal(sum, gen_ext_sum(block)));
  test_assert(equal(sum, gen_blk_sum(block)));
  test_assert(equal(sum, gen_ext_sum(pb)));
  test_assert(equal(sum, gen_blk_sum(pb)));
}



/// Test low-level data interface to 1-dimensional blocks.

/// Requires:
///   LLDI is a low-level data interface, such as
///      vsip::impl::data_interface::Direct_data, or
///      vsip::impl::data_interface::Copy_data.
///
template <typename Block,
          typename AccessTag>
void
test_1_low()
{
  length_type const size = 10;

  typedef typename Block::value_type value_type;

  Block block(size, 0.0);

  value_type val0 =  1.0f;
  value_type val1 =  2.78f;
  value_type val2 =  3.14f;
  value_type val3 = -1.5f;

  // Place values in block.
  block.put(0, val0);
  block.put(1, val1);

  {
    vsip::dda::impl::Accessor<Block,
      typename Block::layout_type,
      AccessTag>
      raw(block);

    // Check properties of LLDI.
    test_assert(raw.stride(&block, 0) == 1);
    test_assert(raw.size(&block, 0) == size);

    float* data = raw.ptr(&block);
    raw.begin(&block, true);

    // Check that block values are reflected.
    test_assert(equal(data[0], val0));
    test_assert(equal(data[1], val1));

    // Place values in raw data.
    data[1] = val2;
    data[2] = val3;

    raw.end(&block, true);
  }

  // Check that raw data values are reflected.
  test_assert(equal(block.get(1), val2));
  test_assert(equal(block.get(2), val3));
}



/// Test high-level data interface to 1-dimensional blocks.

template <typename Block>
void
test_1_ext()
{
  length_type const size = 10;

  typedef typename Block::value_type value_type;

  Block block(size, 0.0);

  value_type val0 =  1.0f;
  value_type val1 =  2.78f;
  value_type val2 =  3.14f;
  value_type val3 = -1.5f;

  // Place values in block.
  block.put(0, val0);
  block.put(1, val1);

  {
    vsip::dda::Data<Block, dda::inout> raw(block);

    // Check properties of DDI.
    test_assert(raw.stride(0) == 1);
    test_assert(raw.size(0) == size);

    float* data = raw.ptr();

    // Check that block values are reflected.
    test_assert(equal(data[0], val0));
    test_assert(equal(data[1], val1));

    // Place values in raw data.
    data[1] = val2;
    data[2] = val3;
  }

  // Check that raw data values are reflected.
  test_assert(equal(block.get(1), val2));
  test_assert(equal(block.get(2), val3));
}



// Test high-level data interface to 2-dim Dense.

void
test_dense_2()
{
  length_type const num_rows = 10;
  length_type const num_cols = 15;

  typedef Dense<2, float, row2_type> Row_major_block;
  typedef Dense<2, float, col2_type> Col_major_block;

  Row_major_block row_blk(Domain<2>(num_rows, num_cols), 0.0);
  Col_major_block col_blk(Domain<2>(num_rows, num_cols), 0.0);

  vsip::dda::Data<Row_major_block, dda::inout> row_raw(row_blk);
  vsip::dda::Data<Col_major_block, dda::inout> col_raw(col_blk);

  test_assert(row_raw.stride(0) == static_cast<stride_type>(num_cols));
  test_assert(row_raw.stride(1) == 1U);
  test_assert(row_raw.size(0) == num_rows);
  test_assert(row_raw.size(1) == num_cols);

  test_assert(col_raw.stride(0) == 1U);
  test_assert(col_raw.stride(1) == static_cast<stride_type>(num_rows));
  test_assert(col_raw.size(0) == num_rows);
  test_assert(col_raw.size(1) == num_cols);
}



// Test that 1-dimensional and 2-dimensional data interfaces to Dense
// are consistent.

template <typename T,
	  typename Order>
void
test_dense_12()
{
  using vsip::Layout;
  using vsip::dda::in;

  length_type const num_rows = 10;
  length_type const num_cols = 15;

  typedef Dense<2, T, Order> Block;

  Block block(Domain<2>(num_rows, num_cols));

  // place values into block for comparison.
  for (index_type r=0; r<num_rows; ++r)
    for (index_type c=0; c<num_cols; ++c)
      block.put(r, c, T(r*num_cols+c));

  // Check 2-dimensional data access.
  {
    vsip::dda::Data<Block, dda::in, Layout<2, Order,
      Block::packing,
      Block::storage_format> > raw(block);

    test_assert(raw.stride(Order::Dim0) == 
	   static_cast<stride_type>(Order::Dim0 == 0 ? num_cols : num_rows));
    test_assert(raw.stride(Order::Dim1) == 1);

    test_assert(raw.size(0) == num_rows);
    test_assert(raw.size(1) == num_cols);

    // Cost should be zero:
    //  - Block is Dense, supports Direct_data,
    //  - Requested layout is same as blocks.
    test_assert(raw.CT_cost == 0);

    for (index_type r=0; r<raw.size(0); ++r)
      for (index_type c=0; c<raw.size(1); ++c)
	test_assert(equal(raw.ptr()[r*raw.stride(0) + c*raw.stride(1)],
		     T(r*num_cols + c)));
  }



  // Check 1-dimensional data access.
  {
    vsip::dda::Data<Block, dda::in, Layout<1, row1_type,
      Block::packing,
      Block::storage_format> > raw(block);

    test_assert(raw.stride(0) == 1);

    test_assert(raw.size(0) == num_rows*num_cols);

    // Cost should be zero:
    //  - Block is Dense, supports Direct_data,
    //  - Requested 1-dim layout is supported.
    test_assert(raw.CT_cost == 0);

    for (index_type r=0; r<num_rows; ++r)
    {
      for (index_type c=0; c<num_cols; ++c)
      {
	index_type idx = (Order::Dim0 == 0)
	            ? (r * num_cols + c)	// row-major
	            : (r + c * num_rows);	// col-major

	test_assert(equal(raw.ptr()[idx], T(r*num_cols + c)));
      }
    }
  }
}



/// Test view sum and dot-product functions.

void
test_view_functions()
{
  length_type size = 10;
  Vector<float> view1(size);
  Vector<float> view2(size);
  
  for (index_type i=0; i<size; ++i)
  {
    view1.put(i, float(i+1));
    view2.put(i, float(2*i+1));
  }

  test_assert(equal(sum_view(view1), sum_ext(view1)));
  test_assert(equal(sum_view(view2), sum_ext(view2)));

  float prod_v = dotp_view(view1, view2);
  float prod_e  = dotp_ext(view1, view2);

  test_assert(equal(prod_v, prod_e));
}



/// Test vector_add function.

void
test_vector_add()
{
  length_type size = 10;
  Vector<float> view1(size);
  Vector<float> view2(size);
  Vector<float> res(size);
  
  for (index_type i=0; i<size; ++i)
  {
    view1.put(i, float(i+1));
    view2.put(i, float(2*i+1));
  }

  vector_add(res, view1, view2);

  for (index_type i=0; i<size; ++i)
  {
    test_assert(equal(res.get(i), view1.get(i) + view2.get(i)));
  }
}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  using vsip::dda::impl::Direct_access_tag;
  using vsip::dda::impl::Copy_access_tag;

  test_block_sum();

  // Test low-level data interface.
  test_1_low<Dense<1, float>, Direct_access_tag>();
  test_1_low<Dense<1, float>, Copy_access_tag>();

  // test_1_low<Strided<1, float>, vsip::impl::Ref_count_policy, Direct_data>();
  test_1_low<vsip::impl::Strided<1, float>, Copy_access_tag>();

  // Test high-level data interface.
  test_1_ext<Dense<1,      float> >();
  test_1_ext<impl::Strided<1, float> >();

  test_dense_2();

  // Test 1-dim direct data views of N-dim blocks.
  // test_dense_12<float, row2_type>();
  // test_dense_12<float, col2_type>();
  // test_dense_12<int,   row2_type>();
  // test_dense_12<int,   col2_type>();
  // test_dense_12<short, row2_type>();
  // test_dense_12<short, col2_type>();

  test_view_functions();

  test_vector_add();
}
