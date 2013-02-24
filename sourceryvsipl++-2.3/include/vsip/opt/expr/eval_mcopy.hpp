/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/mcopy.hpp
    @author  Jules Bergmann
    @date    2008-09-19
    @brief   VSIPL++ Library: Generic matrix copy/transpose evaluator.
*/

#ifndef VSIP_OPT_EXPR_MCOPY_HPP
#define VSIP_OPT_EXPR_MCOPY_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/fast_transpose.hpp>

namespace vsip
{
namespace impl
{

/// Generic in-place transpose
template <typename T>
void
ip_transpose(T *d_ptr,
	     stride_type row_stride,
	     stride_type col_stride,
	     length_type rows,
	     length_type cols)
{
  for (index_type i=0; i<rows; ++i)
    for (index_type j=i; j<cols; ++j)
    {
      std::swap(d_ptr[col_stride * i + row_stride * j],
		d_ptr[col_stride * j + row_stride * i]);
    }
}

/// Split-complex in-place transpose
template <typename T>
void
ip_transpose(std::pair<T*, T*> const &d,
	     stride_type row_stride,
	     stride_type col_stride,
	     length_type rows,
	     length_type cols)
{
  ip_transpose(d.first,  row_stride, col_stride, rows, cols);
  ip_transpose(d.second, row_stride, col_stride, rows, cols);
}

/// Generic matrix copy.
template <typename T>
void
mcopy(T *d_ptr,
      T *s_ptr,
      stride_type d_stride_0,
      stride_type d_stride_1,
      stride_type s_stride_0,
      stride_type s_stride_1,
      length_type size_0,
      length_type size_1)
{
  if (d_stride_1 == 1 && s_stride_1 == 1)
  {
    // Dense
    if (d_stride_0 == static_cast<stride_type>(size_1) &&
	s_stride_0 == static_cast<stride_type>(size_1))
      memcpy(d_ptr, s_ptr, size_0*size_1*sizeof(T));
    // Contiguous rows
    else
      for (index_type i=0; i<size_0; ++i)
      {
	memcpy(d_ptr, s_ptr, size_1*sizeof(T));
	d_ptr += d_stride_0;
	s_ptr += s_stride_0;
      }
  }
  else
  {
    for (index_type i=0; i<size_0; ++i)
    {
      T* d_row = d_ptr;
      T* s_row = s_ptr;
      
      for (index_type j=0; j<size_1; ++j)
      {
	*d_row = *s_row;
	d_row += d_stride_1;
	s_row += s_stride_1;
      }
      
      d_ptr += d_stride_0;
      s_ptr += s_stride_0;
    }
  }
}

/// Split-complex matrix copy.
template <typename T>
void
mcopy(std::pair<T*, T*> const &d,
      std::pair<T*, T*> const &s,
      stride_type d_stride_0,
      stride_type d_stride_1,
      stride_type s_stride_0,
      stride_type s_stride_1,
      length_type size_0,
      length_type size_1)
{
  mcopy(d.first, s.first,
	d_stride_0, d_stride_1,
	s_stride_0, s_stride_1,
	size_0, size_1);
  mcopy(d.second, s.second,
	d_stride_0, d_stride_1,
	s_stride_0, s_stride_1,
	size_0, size_1);
}
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
/// Matrix copy and transpose evaluator.
template <typename DstBlock, typename SrcBlock>
struct Evaluator<op::assign<2>, be::transpose,
		 void(DstBlock &, SrcBlock const &)>
{
  static char const* name()
  {
    char s = impl::Type_equal<src_order_type, row2_type>::value ? 'r' : 'c';
    char d = impl::Type_equal<dst_order_type, row2_type>::value ? 'r' : 'c';
    if      (s == 'r' && d == 'r')    return "Expr_Trans (rr copy)";
    else if (s == 'r' && d == 'c')    return "Expr_Trans (rc trans)";
    else if (s == 'c' && d == 'r')    return "Expr_Trans (cr trans)";
    else /* (s == 'c' && d == 'c') */ return "Expr_Trans (cc copy)";
  }

  typedef typename DstBlock::value_type dst_value_type;
  typedef typename SrcBlock::value_type src_value_type;

  static bool const is_rhs_expr   = impl::Is_expr_block<SrcBlock>::value;

  static bool const is_rhs_simple =
    impl::Is_simple_distributed_block<SrcBlock>::value;

  static bool const is_lhs_split  = impl::Is_split_block<DstBlock>::value;
  static bool const is_rhs_split  = impl::Is_split_block<SrcBlock>::value;

  static int const  lhs_cost      = impl::Ext_data_cost<DstBlock>::value;
  static int const  rhs_cost      = impl::Ext_data_cost<SrcBlock>::value;

  typedef typename impl::Block_layout<SrcBlock>::order_type src_order_type;
  typedef typename impl::Block_layout<DstBlock>::order_type dst_order_type;

  static bool const ct_valid =
    impl::Type_equal<src_value_type, dst_value_type>::value &&
    !is_rhs_expr &&
    lhs_cost == 0 && rhs_cost == 0 &&
    (is_lhs_split == is_rhs_split);

  static bool rt_valid(DstBlock& /*dst*/, SrcBlock const& /*src*/)
  { return true; }

  static void exec(DstBlock& dst, SrcBlock const& src, row2_type, row2_type)
  {
    vsip::impl::Ext_data<DstBlock> d_ext(dst, vsip::impl::SYNC_OUT);
    vsip::impl::Ext_data<SrcBlock> s_ext(src, vsip::impl::SYNC_IN);

    impl::mcopy(d_ext.data(), s_ext.data(),
		d_ext.stride(0), d_ext.stride(1),
		s_ext.stride(0), s_ext.stride(1),
		d_ext.size(0), d_ext.size(1));
  }

  static void exec(DstBlock& dst, SrcBlock const& src, col2_type, col2_type)
  {
    vsip::impl::Ext_data<DstBlock> d_ext(dst, vsip::impl::SYNC_OUT);
    vsip::impl::Ext_data<SrcBlock> s_ext(src, vsip::impl::SYNC_IN);

    impl::mcopy(d_ext.data(), s_ext.data(),
		d_ext.stride(1), d_ext.stride(0),
		s_ext.stride(1), s_ext.stride(0),
		d_ext.size(1), d_ext.size(0));
  }

  static void exec(DstBlock& dst, SrcBlock const& src, col2_type, row2_type)
  {
    vsip::impl::Ext_data<DstBlock> dst_ext(dst, vsip::impl::SYNC_OUT);
    vsip::impl::Ext_data<SrcBlock> src_ext(src, vsip::impl::SYNC_IN);

    // In-place transpose
    if (impl::is_same_ptr(dst_ext.data(), src_ext.data()))
    {
      // in-place transpose implies square matrix
      assert(dst.size(2, 0) == dst.size(2, 1));
      impl::ip_transpose(dst_ext.data(),
			 dst_ext.stride(0), dst_ext.stride(1),
			 dst.size(2, 0), dst.size(2, 1));
    }
    else if (dst_ext.stride(0) == 1 && src_ext.stride(1) == 1)
    {
      impl::transpose_unit(dst_ext.data(), src_ext.data(),
			   dst.size(2, 0), dst.size(2, 1), // rows, cols
			   dst_ext.stride(1),	           // dst_col_stride
			   src_ext.stride(0));	           // src_row_stride
    }
    else
    {
      impl::transpose(dst_ext.data(), src_ext.data(),
		      dst.size(2, 0), dst.size(2, 1),		// rows, cols
		      dst_ext.stride(0), dst_ext.stride(1),	// dst strides
		      src_ext.stride(0), src_ext.stride(1));	// srd strides
    }
  }

  static void exec(DstBlock& dst, SrcBlock const& src, row2_type, col2_type)
  {
    vsip::impl::Ext_data<DstBlock> dst_ext(dst, vsip::impl::SYNC_OUT);
    vsip::impl::Ext_data<SrcBlock> src_ext(src, vsip::impl::SYNC_IN);

    // In-place transpose
    if (impl::is_same_ptr(dst_ext.data(), src_ext.data()))
    {
      // in-place transpose implies square matrix
      assert(dst.size(2, 0) == dst.size(2, 1));
      impl::ip_transpose(dst_ext.data(),
			 dst_ext.stride(0), dst_ext.stride(1),
			 dst.size(2, 0), dst.size(2, 1));
    }
    else if (dst_ext.stride(1) == 1 && src_ext.stride(0) == 1)
    {
      impl::transpose_unit(dst_ext.data(), src_ext.data(),
			   dst.size(2, 1), dst.size(2, 0), // rows, cols
			   dst_ext.stride(0),	  // dst_col_stride
			   src_ext.stride(1));	  // src_row_stride
    }
    else
    {
      impl::transpose(dst_ext.data(), src_ext.data(),
		      dst.size(2, 1), dst.size(2, 0), // rows, cols
		      dst_ext.stride(1), dst_ext.stride(0),	// dst strides
		      src_ext.stride(1), src_ext.stride(0));	// srd strides
    }
  }

  static void exec(DstBlock& dst, SrcBlock const& src)
  {
    exec(dst, src, dst_order_type(), src_order_type());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
