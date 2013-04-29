/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL++ Library: Unit tests for runtime determination of
///   direct versus copy access.

#include <iostream>
#include <cassert>
#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>

#include <vsip_csl/test.hpp>
#include "output.hpp"

#define VERBOSE 1

using namespace vsip;
using vsip_csl::equal;

using vsip::impl::conditional;
using vsip::impl::is_same;

length_type g_size = 10;

length_type g_rows = 10;
length_type g_cols = 15;

length_type g_dim0 = 8;
length_type g_dim1 = 12;
length_type g_dim2 = 14;



/***********************************************************************
  Definitions
***********************************************************************/

template <typename BlockT>
void
dump_access_details()
{
  std::cout << "Access details (block_type = " << Type_name<BlockT>::name() << std::endl;

  typedef vsip::dda::dda_block_layout<BlockT> dbl_type;
  typedef typename dbl_type::access_type access_type;
  typedef typename dbl_type::order_type  order_type;
  static pack_type const packing = dbl_type::packing;
  typedef typename dbl_type::layout_type layout_type;

  std::cout << "  dbl access_type = " << Type_name<access_type>::name() << std::endl;
  std::cout << "  dbl order_type  = " << Type_name<order_type>::name() << std::endl;
  std::cout << "  dbl packing   = " << packing << std::endl;

  typedef vsip::get_block_layout<BlockT> bl_type;
  typedef typename bl_type::access_type bl_access_type;
  typedef typename bl_type::order_type  bl_order_type;
  static pack_type const bl_packing = bl_type::packing;
  typedef typename bl_type::layout_type bl_layout_type;

  std::cout << "  bl  access_type = " << Type_name<bl_access_type>::name() << std::endl;
  std::cout << "  bl  order_type  = " << Type_name<bl_order_type>::name() << std::endl;
  std::cout << "  bl  packing   = " << bl_packing << std::endl;

  typedef typename vsip::dda::impl::Choose_access<BlockT, layout_type>::type
    use_access_type;

  std::cout << "  use_access_type = " << Type_name<use_access_type>::name() << std::endl;

  std::cout << "  cost            = " << vsip::dda::Data<BlockT, dda::in>::ct_cost
       << std::endl;
}


struct Same {};
struct Diff {};

struct Full {};		// full dimension
struct Cont {};		// continuous - first half of dimension
struct Off1 {};		// ^ ^ offset by 1
struct Spar {};		// sparse - every other element
struct Spar4 {};		// sparse - every other element
struct Sing {};		// single - one element



/***********************************************************************
  Vector test harnass
***********************************************************************/

template <pack_type P,
	  storage_format_type C,
	  typename T,
	  typename BlockT>
void
test_vector(
  int               ct_cost,
  int               rt_cost,
  Vector<T, BlockT> view)
{
  // length_type size = view.size(0);

  dimension_type const dim = BlockT::dim;
  typedef typename impl::conditional<supports_dda<BlockT>::value,
    dda::impl::Direct_access_tag, dda::impl::Copy_access_tag>::type access_type;
  typedef typename get_block_layout<BlockT>::order_type   order_type;
  static pack_type const blk_packing = get_block_layout<BlockT>::packing;

  typedef vsip::Layout<dim, order_type, P, C> LP;

  typedef typename vsip::dda::impl::Choose_access<BlockT, LP>::type real_access_type;

  for (index_type i=0; i<view.size(0); ++i)
    view.put(i, T(i));

  {
    vsip::dda::Data<BlockT, dda::inout, LP> data(view.block());

#if VERBOSE
    std::cout << "Block (" << Type_name<BlockT>::name() << ")" << std::endl
	 << "  Blk AT   = " << Type_name<access_type>::name() << std::endl
	 << "  Blk pack = " << blk_packing << std::endl

	 << "  Req AT   = " << Type_name<real_access_type>::name() << std::endl
	 << "  Req pack = " << P << std::endl
	 << "  ct_cost  = " << vsip::dda::Data<BlockT, dda::in, LP>::ct_cost << std::endl
	 << "  rt_cost  = " << data.cost() << std::endl
      ;
#endif
    
    test_assert((ct_cost == vsip::dda::Data<BlockT, dda::in, LP>::ct_cost));
    test_assert(rt_cost == data.cost());
    test_assert(rt_cost == vsip::dda::cost<LP>(view.block()));

    // Check that rt_cost == 0 implies mem_required == 0
    test_assert((rt_cost == 0 && vsip::dda::impl::mem_required<LP>(view.block()) == 0) ||
		(rt_cost != 0 && vsip::dda::impl::mem_required<LP>(view.block()) > 0));

    // Check that rt_cost == 0 implies xfer_required == false
    test_assert((rt_cost == 0 && !vsip::dda::impl::xfer_required<LP>(view.block())) ||
		(rt_cost != 0 &&  vsip::dda::impl::xfer_required<LP>(view.block())) );

    test_assert(data.size(0) == view.size(0));

    T *ptr = data.ptr();
    stride_type stride0 = data.stride(0);

    for (index_type i=0; i<data.size(0); ++i)
    {
      test_assert(equal(ptr[i*stride0], T(i)));
      ptr[i*stride0] = T(i+100);
    }
  }

  for (index_type i=0; i<view.size(0); ++i)
    test_assert(equal(view.get(i), T(i+100)));
}



template <typename BlockT,
	  typename Dim0,
	  pack_type P,
	  storage_format_type C>
struct Test_vector
{
  static void
  test(int ct_cost, int rt_cost)
  {
    Vector<typename BlockT::value_type, BlockT> view(g_size);

    typedef Domain<1> D1;

    D1 dom0(view.size(0));

    if      (is_same<Dim0, Cont>::value) dom0 = D1(0, 1, view.size(0)/2);
    else if (is_same<Dim0, Off1>::value) dom0 = D1(1, 1, view.size(0)/2);
    else if (is_same<Dim0, Spar>::value) dom0 = D1(0, 2, view.size(0)/2);

    test_vector<P, C>(ct_cost, rt_cost, view(dom0));
  }
};


template <typename BlockT,
	  pack_type P,
	  storage_format_type C>
struct Test_vector<BlockT, Full, P, C>
{
  static void
  test(int ct_cost, int rt_cost)
  {
    Vector<typename BlockT::value_type, BlockT> view(g_size);

    test_vector<P, C>(ct_cost, rt_cost, view);
  }
};



template <typename Block,
	  typename D,
	  pack_type ReqP,
	  bool Same,
	  storage_format_type BlkC = vsip::get_block_layout<Block>::storage_format,
	  pack_type const ActP = ReqP,
	  storage_format_type const ActC = Same ? BlkC :
	  (BlkC == split_complex ? interleaved_complex : split_complex)>
struct Tv : Test_vector<Block, D, ActP, ActC>
{};



/***********************************************************************
  Matrix test harnass
***********************************************************************/

template <pack_type P,
	  typename OrderT,
	  storage_format_type C,
	  typename T,
	  typename BlockT>
void
test_matrix(int ct_cost,
	    int rt_cost,
	    Matrix<T, BlockT> view)
{
  // length_type size = view.size(0);

  dimension_type const dim = BlockT::dim;
  typedef typename impl::conditional<supports_dda<BlockT>::value,
    dda::impl::Direct_access_tag, dda::impl::Copy_access_tag>::type access_type;
  typedef OrderT  order_type;

  typedef vsip::Layout<dim, order_type, P, C> LP;

  typedef typename vsip::dda::impl::Choose_access<BlockT, LP>::type real_access_type;

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      view.put(i, j, T(i*view.size(1)+j));

  {
    vsip::dda::Data<BlockT, dda::inout, LP> data(view.block());

#if VERBOSE
    std::cout << "Block (" << Type_name<BlockT>::name() << ")" << std::endl
	 << "  AT      = " << Type_name<access_type>::name() << std::endl
	 << "  RAT     = " << Type_name<real_access_type>::name() << std::endl
	 << "  ct_cost = " << vsip::dda::Data<BlockT, dda::inout, LP>::ct_cost << std::endl
	 << "  rt_cost = " << data.cost() << std::endl
      ;
#endif
    
    test_assert((ct_cost == vsip::dda::Data<BlockT, dda::inout, LP>::ct_cost));
    test_assert(rt_cost == data.cost());

    // Check that rt_cost == 0 implies mem_required == 0
    test_assert((rt_cost == 0 && vsip::dda::impl::mem_required<LP>(view.block()) == 0) ||
		(rt_cost != 0 && vsip::dda::impl::mem_required<LP>(view.block()) > 0));

    // Check that rt_cost == 0 implies xfer_required == false
    test_assert((rt_cost == 0 && !vsip::dda::impl::xfer_required<LP>(view.block())) ||
		(rt_cost != 0 &&  vsip::dda::impl::xfer_required<LP>(view.block())) );

    test_assert(data.size(0) == view.size(0));
    test_assert(data.size(1) == view.size(1));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    stride_type stride1 = data.stride(1);

    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
      {
	test_assert(equal(ptr[i*stride0 + j*stride1], T(i*view.size(1)+j)));
	ptr[i*stride0 + j*stride1] = T(i+j*view.size(0));
      }
  }

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      test_assert(equal(view.get(i, j), T(i+j*view.size(0))));
}



template <typename BlockT,
	  typename Dim0,
	  typename Dim1,
	  pack_type P,
	  typename OrderT,
	  storage_format_type C>
struct Test_matrix
{
  static void
  test(int ct_cost, int rt_cost)
  {
    Matrix<typename BlockT::value_type, BlockT> view(g_rows, g_cols);

    typedef Domain<1> D1;

    Domain<1> dom0(view.size(0));
    Domain<1> dom1(view.size(1));

    if      (is_same<Dim0, Cont>::value)  dom0 = D1(0, 1, view.size(0)/2);
    else if (is_same<Dim0, Off1>::value)  dom0 = D1(1, 1, view.size(0)/2);
    else if (is_same<Dim0, Spar>::value)  dom0 = D1(0, 2, view.size(0)/2);
    else if (is_same<Dim0, Spar4>::value) dom0 = D1(0, 4, view.size(0)/4);
    else if (is_same<Dim0, Sing>::value)  dom0 = D1(0, 1, 1);

    if      (is_same<Dim1, Cont>::value)  dom1 = D1(0, 1, view.size(1)/2);
    else if (is_same<Dim1, Off1>::value)  dom1 = D1(1, 1, view.size(1)/2);
    else if (is_same<Dim1, Spar>::value)  dom1 = D1(0, 2, view.size(1)/2);
    else if (is_same<Dim1, Spar4>::value) dom1 = D1(0, 4, view.size(1)/4);
    else if (is_same<Dim1, Sing>::value)  dom1 = D1(0, 1, 1);

    Domain<2> dom(dom0, dom1);

    test_matrix<P, OrderT, C>(ct_cost, rt_cost, view(dom));
  }
};


template <typename BlockT,
	  pack_type P,
	  typename OrderT,
	  storage_format_type C>
struct Test_matrix<BlockT, Full, Full, P, OrderT, C>
{
  static void
  test(int ct_cost, int rt_cost)
  {
    Matrix<typename BlockT::value_type, BlockT> view(g_rows, g_cols);

    test_matrix<P, OrderT, C>(ct_cost, rt_cost, view);
  }
};


template <typename Block,
	  typename Dim0,
	  typename Dim1,
	  pack_type ReqP,
	  typename ReqO,
	  bool SameC,
	  typename BlkO = typename vsip::get_block_layout<Block>::order_type,
	  storage_format_type BlkC = vsip::get_block_layout<Block>::storage_format,

	  typename O = typename
	  conditional<is_same<ReqO, Same>::value,
	    BlkO,
	    typename conditional<is_same<ReqO, Diff>::value,
	      typename conditional<is_same<BlkO, row2_type>::value, col2_type, row2_type>::type,
	      ReqO>::type>::type,
	  storage_format_type C = SameC ? BlkC :
	  (BlkC == interleaved_complex ? split_complex : interleaved_complex)>
struct Tm : Test_matrix<Block, Dim0, Dim1, ReqP, O, C>
{};

/***********************************************************************
  Tensor test harnass
***********************************************************************/

template <pack_type P,
	  typename O,
	  storage_format_type C,
	  typename T,
	  typename Block>
void
test_tensor(int ct_cost, int rt_cost, Tensor<T, Block> view)
{
  dimension_type const dim = Block::dim;
  typedef typename impl::conditional<supports_dda<Block>::value,
    dda::impl::Direct_access_tag, dda::impl::Copy_access_tag>::type access_type;
  typedef vsip::Layout<dim, O, P, C> LP;

  typedef typename vsip::dda::impl::Choose_access<Block, LP>::type real_access_type;

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      for (index_type k=0; k<view.size(2); ++k)
	view.put(i, j, k, T(i*view.size(1)*view.size(2) + j*view.size(2) + k));

  {
    vsip::dda::Data<Block, dda::inout, LP> data(view.block());

#if VERBOSE
    std::cout << "Block (" << Type_name<Block>::name() << ")" << std::endl
	 << "  AT      = " << Type_name<access_type>::name() << std::endl
	 << "  RAT     = " << Type_name<real_access_type>::name() << std::endl
	 << "  ct_cost = " << vsip::dda::Data<Block, dda::inout, LP>::ct_cost << std::endl
	 << "  rt_cost = " << data.cost() << std::endl
      ;
#endif
    
    test_assert((ct_cost == vsip::dda::Data<Block, dda::inout, LP>::ct_cost));
    test_assert(rt_cost == data.cost());

    // Check that rt_cost == 0 implies mem_required == 0
    test_assert((rt_cost == 0 && vsip::dda::impl::mem_required<LP>(view.block()) == 0) ||
		(rt_cost != 0 && vsip::dda::impl::mem_required<LP>(view.block()) > 0));

    // Check that rt_cost == 0 implies xfer_required == false
    test_assert((rt_cost == 0 && !vsip::dda::impl::xfer_required<LP>(view.block())) ||
		(rt_cost != 0 &&  vsip::dda::impl::xfer_required<LP>(view.block())) );

    test_assert(data.size(0) == view.size(0));
    test_assert(data.size(1) == view.size(1));
    test_assert(data.size(2) == view.size(2));

    T* ptr              = data.ptr();
    stride_type stride0 = data.stride(0);
    stride_type stride1 = data.stride(1);
    stride_type stride2 = data.stride(2);

    for (index_type i=0; i<data.size(0); ++i)
      for (index_type j=0; j<data.size(1); ++j)
	for (index_type k=0; k<data.size(2); ++k)
	{
	  test_assert(equal(ptr[i*stride0 + j*stride1 + k*stride2],
		       T(i*view.size(1)*view.size(2) + j*view.size(2) + k)));
	  ptr[i*stride0 + j*stride1 + k*stride2] =
	    T(i+j*view.size(0)+k*view.size(0)*view.size(1));
      }
  }

  for (index_type i=0; i<view.size(0); ++i)
    for (index_type j=0; j<view.size(1); ++j)
      for (index_type k=0; k<view.size(2); ++k)
	test_assert(equal(view.get(i, j, k),
		     T(i+j*view.size(0)+k*view.size(0)*view.size(1))));
}



template <typename Block,
	  typename Dim0,
	  typename Dim1,
	  typename Dim2,
	  pack_type P,
	  typename O,
	  storage_format_type C>
struct Test_tensor
{
  static void
  test(int ct_cost, int rt_cost)
  {
    Tensor<typename Block::value_type, Block> view(g_dim0, g_dim1, g_dim2);

    typedef Domain<1> D1;

    Domain<1> dom0(view.size(0));
    Domain<1> dom1(view.size(1));
    Domain<1> dom2(view.size(2));

    if      (is_same<Dim0, Cont>::value) dom0 = D1(0, 1, view.size(0)/2);
    else if (is_same<Dim0, Off1>::value) dom0 = D1(1, 1, view.size(0)/2);
    else if (is_same<Dim0, Spar>::value) dom0 = D1(0, 2, view.size(0)/2);
    else if (is_same<Dim0, Sing>::value) dom0 = D1(0, 1, 1);

    if      (is_same<Dim1, Cont>::value) dom1 = D1(0, 1, view.size(1)/2);
    else if (is_same<Dim1, Off1>::value) dom1 = D1(1, 1, view.size(1)/2);
    else if (is_same<Dim1, Spar>::value) dom1 = D1(0, 2, view.size(1)/2);
    else if (is_same<Dim1, Sing>::value) dom1 = D1(1, 1, 1);

    if      (is_same<Dim2, Cont>::value) dom2 = D1(0, 1, view.size(2)/2);
    else if (is_same<Dim2, Off1>::value) dom2 = D1(1, 1, view.size(2)/2);
    else if (is_same<Dim2, Spar>::value) dom2 = D1(0, 2, view.size(2)/2);
    else if (is_same<Dim2, Sing>::value) dom2 = D1(1, 1, 1);

    Domain<3> dom(dom0, dom1, dom2);

    test_tensor<P, O, C>(ct_cost, rt_cost, view(dom));
  }
};


template <typename Block, pack_type P, typename O, storage_format_type C>
struct Test_tensor<Block, Full, Full, Full, P, O, C>
{
  static void
  test(int ct_cost, int rt_cost)
  {
    Tensor<typename Block::value_type, Block> view(g_dim0, g_dim1, g_dim2);
    test_tensor<P, O, C>(ct_cost, rt_cost, view);
  }
};



template <typename BlockT,
	  typename Dim0,
	  typename Dim1,
	  typename Dim2,
	  pack_type P,
	  typename ReqO,
	  bool SameC,
	  typename BlkO = typename vsip::get_block_layout<BlockT>::order_type,
	  storage_format_type BlkC = vsip::get_block_layout<BlockT>::storage_format,
	  typename O = typename
	  conditional<is_same<ReqO, Same>::value,
	    BlkO,
	    typename conditional<is_same<ReqO, Diff>::value,
	      typename conditional<is_same<BlkO, row2_type>::value,
		col2_type, row2_type>::type,
		ReqO>::type>::type,

	  storage_format_type C = SameC ? BlkC :
	  (BlkC == interleaved_complex ? split_complex : interleaved_complex)>
struct Tt : Test_tensor<BlockT, Dim0, Dim1, Dim2, P, O, C>
{};

/***********************************************************************
  Tests
***********************************************************************/

void
vector_tests()
{
  typedef Dense<1, float> block_type;

  // Blk  , Dim0, Pack          , Cplx
  // -----,-----,---------------,-----

  // Asking for any_packing packing indicates we don't care.
  //   If we ask for same complex format => get direct access
  //   If we ask for different complex format => get copy access
  Tv<block_type, Full, any_packing, true>::test(0, 0);
  Tv<block_type, Cont, any_packing, true>::test(0, 0);
  Tv<block_type, Spar, any_packing, true>::test(0, 0);
  Tv<block_type, Full, any_packing, false>::test(2, 2);
  Tv<block_type, Cont, any_packing, false>::test(2, 2);
  Tv<block_type, Spar, any_packing, false>::test(2, 2);


  // Asking for packing::unit_stride packing indicates we want unit stride,
  //   (but don't care about overall dense-ness or alignment)
  // - Subviews will have a compile-time cost of 2, but a runtime cost
  //   of 0 iff they have stride-1.
  // - Different complex format => copy access
  Tv<block_type, Full, unit_stride, true>::test(0, 0);
  Tv<block_type, Cont, unit_stride, true>::test(2, 0);	// runtime stride-1
  Tv<block_type, Spar, unit_stride, true>::test(2, 2);
  Tv<block_type, Full, unit_stride, false>::test(2, 2);
  Tv<block_type, Cont, unit_stride, false>::test(2, 2);
  Tv<block_type, Spar, unit_stride, false>::test(2, 2);

  Tv<block_type, Full, dense, true>::test(0, 0);
  Tv<block_type, Cont, dense, true>::test(2, 0);
  Tv<block_type, Off1, dense, true>::test(2, 0);
  Tv<block_type, Spar, dense, true>::test(2, 2);
  Tv<block_type, Full, dense, false>::test(2, 2);
  Tv<block_type, Cont, dense, false>::test(2, 2);
  Tv<block_type, Spar, dense, false>::test(2, 2);

  Tv<block_type, Full, aligned_16, true>::test(2, 0);
  Tv<block_type, Cont, aligned_16, true>::test(2, 0);
  Tv<block_type, Off1, aligned_16, true>::test(2, 2);
  Tv<block_type, Spar, aligned_16, true>::test(2, 2);
}


void
matrix_tests()
{
  typedef Dense<2, float, row2_type> row_type;
  typedef Dense<2, float, col2_type> col_type;

  Tm<row_type, Full, Full, dense, Same, true>::test(0, 0);
  Tm<col_type, Full, Full, dense, Same, true>::test(0, 0);


  // Rationale: By asking for something with any_packing, we should
  //   get the lowest cost at both compile-time and run-time.  Since
  //   Dense is direct, cost should be 0.

  Tm<row_type, Full, Full, any_packing, Same, true>::test(0, 0);
  Tm<row_type, Cont, Cont, any_packing, Same, true>::test(0, 0);
  Tm<row_type, Spar, Spar, any_packing, Same, true>::test(0, 0);


  // Rationale: Asking for stride_unknown but with a particular
  //   dimension-order indicates that dimension-order is
  //   most important, only copy to rearrange.

  Tm<row_type, Full, Full, any_packing, row2_type, true>::test(0, 0);
  Tm<row_type, Full, Full, any_packing, col2_type, true>::test(2, 2);


  // Rationale: Asking for stride_unit, we should get compile-time
  //   direct access for whole views and run-time direct access
  //   if the subview has unit-stride in the lowest order dimension.

  Tm<row_type, Full, Full, unit_stride, Same, true>::test(0, 0);
  Tm<row_type, Cont, Cont, unit_stride, Same, true>::test(2, 0);
  Tm<row_type, Cont, Spar, unit_stride, Same, true>::test(2, 2);
  Tm<row_type, Spar, Cont, unit_stride, Same, true>::test(2, 0);
  Tm<row_type, Spar, Spar, unit_stride, Same, true>::test(2, 2);

  Tm<col_type, Full, Full, unit_stride, Same, true>::test(0, 0);
  Tm<col_type, Cont, Cont, unit_stride, Same, true>::test(2, 0);
  Tm<col_type, Cont, Spar, unit_stride, Same, true>::test(2, 0);
  Tm<col_type, Spar, Cont, unit_stride, Same, true>::test(2, 2);
  Tm<col_type, Spar, Spar, unit_stride, Same, true>::test(2, 2);


  // Rationale: Asking for stride_unit but with a particular
  //   dimension-order indicates that we should copy if either
  //   the dimension-order is wrong, or the lowest-dimension
  //   is not unit stride.

  Tm<row_type, Full, Full, unit_stride, row2_type, true>::test(0, 0);
  Tm<row_type, Full, Cont, unit_stride, row2_type, true>::test(2, 0);
  Tm<row_type, Full, Spar, unit_stride, row2_type, true>::test(2, 2);
  Tm<row_type, Full, Full, unit_stride, col2_type, true>::test(2, 2);
  Tm<row_type, Cont, Full, unit_stride, col2_type, true>::test(2, 2);
  Tm<row_type, Spar, Full, unit_stride, col2_type, true>::test(2, 2);

  Tm<col_type, Full, Full, unit_stride, row2_type, true>::test(2, 2);
  Tm<col_type, Full, Cont, unit_stride, row2_type, true>::test(2, 2);
  Tm<col_type, Full, Spar, unit_stride, row2_type, true>::test(2, 2);
  Tm<col_type, Full, Full, unit_stride, col2_type, true>::test(0, 0);
  Tm<col_type, Cont, Full, unit_stride, col2_type, true>::test(2, 0);
  Tm<col_type, Spar, Full, unit_stride, col2_type, true>::test(2, 2);


  // Rationale: Asking for dense
  //   compile-time cost of 0 iff whole-view
  //   run-time cost of 0 if:
  //    - dimension order subdomains: Single* Cont? Full*, AND
  //    - order is same, complex is same

  // Blk  , Dim0, Dim1, Pack             , Ord , Cplx
  // -----,-----,-----,------------------,-----,-------------------
  Tm<row_type, Full, Full, dense, Same, true>::test(0, 0);
  Tm<row_type, Cont, Cont, dense, Same, true>::test(2, 2);
  Tm<row_type, Cont, Full, dense, Same, true>::test(2, 0);
  Tm<row_type, Full, Cont, dense, Same, true>::test(2, 2);
  Tm<row_type, Sing, Full, dense, Same, true>::test(2, 0);
  Tm<row_type, Sing, Cont, dense, Same, true>::test(2, 0);
  Tm<row_type, Full, Sing, dense, Same, true>::test(2, 2);
  Tm<row_type, Full, Sing, dense, Same, true>::test(2, 2);

  Tm<row_type, Full, Full, dense, Diff, true>::test(2, 2);
  Tm<row_type, Cont, Cont, dense, Diff, true>::test(2, 2);
  Tm<row_type, Cont, Full, dense, Diff, true>::test(2, 2);
  Tm<row_type, Full, Cont, dense, Diff, true>::test(2, 2);
  Tm<row_type, Sing, Full, dense, Diff, true>::test(2, 2);
  Tm<row_type, Sing, Cont, dense, Diff, true>::test(2, 2);
  Tm<row_type, Full, Sing, dense, Diff, true>::test(2, 2);
  Tm<row_type, Full, Sing, dense, Diff, true>::test(2, 2);

  Tm<col_type, Full, Full, dense, Same, true>::test(0, 0);
  Tm<col_type, Cont, Cont, dense, Same, true>::test(2, 2);
  Tm<col_type, Cont, Full, dense, Same, true>::test(2, 2);
  Tm<col_type, Full, Cont, dense, Same, true>::test(2, 0);
  Tm<col_type, Sing, Full, dense, Same, true>::test(2, 2);
  Tm<col_type, Sing, Cont, dense, Same, true>::test(2, 2);
  Tm<col_type, Full, Sing, dense, Same, true>::test(2, 0);
  Tm<col_type, Full, Sing, dense, Same, true>::test(2, 0);

  Tm<col_type, Full, Full, dense, Diff, true>::test(2, 2);
  Tm<col_type, Cont, Cont, dense, Diff, true>::test(2, 2);
  Tm<col_type, Cont, Full, dense, Diff, true>::test(2, 2);
  Tm<col_type, Full, Cont, dense, Diff, true>::test(2, 2);
  Tm<col_type, Sing, Full, dense, Diff, true>::test(2, 2);
  Tm<col_type, Sing, Cont, dense, Diff, true>::test(2, 2);
  Tm<col_type, Full, Sing, dense, Diff, true>::test(2, 2);
  Tm<col_type, Full, Sing, dense, Diff, true>::test(2, 2);


  // Rationale: Dense blocks are not currently aligned at
  // compile-time.

  // Row-major is not aligned.
  test_assert(g_cols * sizeof(float) % 16 != 0);
  Tm<row_type, Full, Full, aligned_16, Same, true>::test(2, 2);

  // Every fourth column is aligned, for row-major:
  test_assert(4 * g_cols * sizeof(float) % 16 == 0);
  Tm<row_type, Spar4, Full, aligned_16, Same, true>::test(2, 0);

  // Column-major is not aligned.
  test_assert(g_rows * sizeof(float) % 16 != 0);
  Tm<col_type, Full, Full, aligned_16, Same, true>::test(2, 2);

  // Every other row is aligned, for column-major:
  test_assert(2 * g_rows * sizeof(float) % 16 == 0);
  Tm<col_type, Full, Spar, aligned_16, Same, true>::test(2, 0);
}

void
tensor_tests()
{
  typedef Dense<3, float, row3_type> row_t;
  typedef Dense<3, float, col3_type> col_t;

  Tt<row_t, Full, Full, Full, any_packing, Same, true>::test(0, 0);
  Tt<row_t, Full, Spar, Cont, any_packing, Same, true>::test(0, 0);
  Tt<row_t, Spar, Spar, Spar, any_packing, Same, true>::test(0, 0);
  
  Tt<row_t, Full, Full, Full, unit_stride, Same, true>::test(0, 0);
  Tt<row_t, Spar, Spar, Full, unit_stride, Same, true>::test(2, 0);
  Tt<row_t, Spar, Spar, Full, unit_stride, Same, true>::test(2, 0);
  Tt<row_t, Spar, Spar, Cont, unit_stride, Same, true>::test(2, 0);
  Tt<row_t, Spar, Spar, Spar, unit_stride, Same, true>::test(2, 2);

  Tt<col_t, Full, Full, Full, unit_stride, Same, true>::test(0, 0);
  Tt<col_t, Full, Spar, Spar, unit_stride, Same, true>::test(2, 0);
  Tt<col_t, Full, Spar, Spar, unit_stride, Same, true>::test(2, 0);
  Tt<col_t, Cont, Spar, Spar, unit_stride, Same, true>::test(2, 0);
  Tt<col_t, Spar, Spar, Spar, unit_stride, Same, true>::test(2, 2);

  Tt<row_t, Full, Full, Full, dense, Same, true>::test(0, 0);
  Tt<row_t, Cont, Full, Full, dense, Same, true>::test(2, 0);
  Tt<row_t, Spar, Full, Full, dense, Same, true>::test(2, 2);
  Tt<row_t, Cont, Cont, Full, dense, Same, true>::test(2, 2);
  Tt<row_t, Sing, Cont, Full, dense, Same, true>::test(2, 0);

  Tt<col_t, Full, Full, Full, dense, Same, true>::test(0, 0);
  Tt<col_t, Full, Full, Cont, dense, Same, true>::test(2, 0);
  Tt<col_t, Full, Full, Spar, dense, Same, true>::test(2, 2);
  Tt<col_t, Full, Cont, Cont, dense, Same, true>::test(2, 2);
  Tt<col_t, Full, Cont, Sing, dense, Same, true>::test(2, 0);

  test_assert(g_dim2 * sizeof(float) % 16 != 0); // row_t is not aligned
  test_assert(g_dim0 * sizeof(float) % 16 == 0); // col_t is     aligned

  Tt<row_t, Full, Full, Full, aligned_16, Same, true>::test(2, 2);
  Tt<col_t, Full, Full, Full, aligned_16, Same, true>::test(2, 0);
  Tt<col_t, Full, Cont, Cont, aligned_16, Same, true>::test(2, 0);
  Tt<col_t, Full, Spar, Spar, aligned_16, Same, true>::test(2, 0);
}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  vector_tests();
  matrix_tests();
  tensor_tests();
}

