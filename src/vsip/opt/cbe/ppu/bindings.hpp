/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/bindings.hpp
    @author  Stefan Seefeld
    @date    2006-12-29
    @brief   VSIPL++ Library: Wrappers and traits to bridge with IBMs CBE SDK.
*/

#ifndef VSIP_OPT_CBE_PPU_BINDINGS_HPP
#define VSIP_OPT_CBE_PPU_BINDINGS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/opt/cbe/vmmul_params.h>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{

template <typename T> 
void
vmmul_row(T const* V, 
	  T const* M, 
	  T*       R, 
	  stride_type m_stride, stride_type r_stride,
	  length_type length, length_type lines, length_type mult);

// Note: this declaration is needed to match the corresponding vmmul_col
// case, though it is currently not implemented.
template <typename T>
void vmmul_row(T const *V,
	       std::pair<T const *, T const *> const& M,
	       std::pair<T*, T*> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length, length_type mult);

template <typename T>
void vmmul_row(std::pair<T const *, T const *> const& V,
	       T const *M,
	       std::pair<T*, T*> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length, length_type mult);

template <typename T>
void vmmul_row(std::pair<T const *, T const *> const& V,
	       std::pair<T const *, T const *> const& M,
	       std::pair<T*, T*> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length, length_type mult);

template <typename T> 
void vmmul_col(T const* V, 
	       T const* M, 
	       T*       R, 
	       stride_type m_stride, stride_type r_stride,
	       length_type length, length_type lines);

template <typename T>
void vmmul_col(T const *V,
	       std::pair<T const *, T const *> const& M,
	       std::pair<T*, T*> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length);

// Note: this declaration is needed to match the corresponding vmmul_row
// case, though it is currently not implemented.
template <typename T>
void vmmul_col(std::pair<T const *, T const *> const& V,
	       T const*                 M,
	       std::pair<T*, T*> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length);
  
template <typename T>
void vmmul_col(std::pair<T const *, T const *> const& V,
	       std::pair<T const *, T const *> const& M,
	       std::pair<T*, T*> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length);



// Vmmul cases supported by CBE backends

template <dimension_type RowCol,
	  typename       VType,
	  bool           VIsSplit,
	  typename       MType,
	  bool           MIsSplit,
	  typename       DstType,
	  bool           DstIsSplit>
struct Is_vmmul_supported
{ static bool const value = false; };

typedef complex<float> CF;

// vmmul_row(C, C, C)
template <>
struct Is_vmmul_supported<row, CF, false, CF, false, CF, false>
{ static bool const value = true; };

// vmmul_row(Z, R, Z)
template <bool NA>  // implies 'not applicable': use for scalar types
struct Is_vmmul_supported<row, CF, true, float, NA, CF, true>
{ static bool const value = true; };

// vmmul_row(Z, Z, Z)
template <>
struct Is_vmmul_supported<row, CF, true, CF, true, CF, true>
{ static bool const value = true; };


// vmmul_col(R, Z, Z)
template <bool NA>
struct Is_vmmul_supported<col, float, NA, CF, true, CF, true>
{ static bool const value = true; };

// vmmul_col(Z, Z, Z)
template <>
struct Is_vmmul_supported<col, CF, true, CF, true, CF, true>
{ static bool const value = true; };

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

/// Evaluator for vector-matrix multiply.

/// Dispatches cases where the dimension order matches the 
/// requested orientation to the SPU's (row-major/by-row and 
/// col-major/by-col).  The other cases are re-dispatched.
template <typename LHS, typename VBlock, typename MBlock, dimension_type D>
struct Evaluator<op::assign<2>, be::cbe_sdk,
		 void(LHS &, expr::Vmmul<D, VBlock, MBlock> const &)>
{
  static char const* name() { return "Cbe_Sdk_Vmmul"; }

  typedef expr::Vmmul<D, VBlock, MBlock> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename VBlock::value_type v_value_type;
  typedef typename MBlock::value_type m_value_type;
  typedef typename get_block_layout<LHS>::type lhs_lp;
  typedef typename get_block_layout<VBlock>::type vblock_lp;
  typedef typename get_block_layout<MBlock>::type mblock_lp;
  typedef typename get_block_layout<LHS>::order_type lhs_order_type;
  typedef typename get_block_layout<MBlock>::order_type src_order_type;

  static bool const is_row_vmmul =
    (D == row && is_same<lhs_order_type, row2_type>::value ||
     D == col && is_same<lhs_order_type, col2_type>::value);

  static bool const ct_valid = 
    !impl::is_expr_block<VBlock>::value &&
    !impl::is_expr_block<MBlock>::value &&
    impl::cbe::Is_vmmul_supported<is_row_vmmul ? row : col,
			    v_value_type,
			    impl::is_split_block<VBlock>::value,
			    m_value_type,
			    impl::is_split_block<MBlock>::value,
			    lhs_value_type,
			    impl::is_split_block<LHS>::value>::value &&
     // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    dda::Data<VBlock, dda::in>::ct_cost == 0 &&
    dda::Data<MBlock, dda::in>::ct_cost == 0;

  static length_type max_vectors_per_group(length_type lines, length_type length)
  {
    length_type mult = 1;
    while ((mult * 2 * length < VSIP_IMPL_MAX_VMMUL_SIZE) &&
           (mult * 2 <= lines))
      mult *= 2;
    return mult;
  }

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    VBlock const& vblock = rhs.get_vblk();
    MBlock const& mblock = rhs.get_mblk();

    dda::Data<LHS, dda::out, lhs_lp> data_dst(lhs);
    dda::Data<VBlock, dda::in, vblock_lp> data_v(vblock);
    dda::Data<MBlock, dda::in, mblock_lp> data_m(mblock);

    typedef typename dda::Data<LHS, dda::out>::element_type lhs_element_type;
    typedef typename dda::Data<MBlock, dda::in>::element_type m_element_type;

    // The value of is_row_vmmul is true when the direction the vector is 
    // oriented for the multiply is along the same lines as the storage order.
    dimension_type const axis = (D == row ? is_row_vmmul : !is_row_vmmul);
    length_type size = lhs.size(2, axis);
    length_type lines = lhs.size(2, 1 - axis);

    if (is_row_vmmul)
    {
      // Check if strides are the right size for transferring one row at a time
      if (impl::cbe::is_dma_stride_ok<lhs_element_type>(data_dst.stride(1 - axis)) &&
          impl::cbe::is_dma_stride_ok<m_element_type>(data_m.stride(1 - axis)))
      {
        return 
          // (large sizes are broken down)
          (size >= VSIP_IMPL_MIN_VMMUL_SIZE) && 
          (data_dst.stride(axis) == 1) &&
          (data_m.stride(axis)   == 1) &&
          (data_v.stride(0) == 1) &&
          impl::cbe::is_dma_addr_ok(data_dst.ptr()) &&
          impl::cbe::is_dma_addr_ok(data_v.ptr()) &&
          impl::cbe::is_dma_addr_ok(data_m.ptr()) &&
          // (non-granular sizes handled)
          impl::cbe::Task_manager::instance()->num_spes() > 0;
      }
      else
      {
        // Determine whether multiple rows (cols) can be grouped together for 
        // efficiency and to better handle odd strides (i.e. four rows/cols of 
        // floats taken at a time will keep the same alignment as the first row).
        // The blocks must be dense in order to use this optimization.  If
        // the lines are too long, then there will be one vector per group.
        length_type mult = max_vectors_per_group(lines, size);
        bool is_input_dense = (data_m.stride(1 - axis) > 0) &&
          (static_cast<length_type>(data_m.stride(1 - axis)) == size);
        bool is_output_dense = (data_m.stride(1 - axis) > 0) &&
          (static_cast<length_type>(data_m.stride(1 - axis)) == size);

        return
          (size >= VSIP_IMPL_MIN_VMMUL_SIZE) && 
          (data_dst.stride(axis) == 1) &&
          (data_m.stride(axis)   == 1) &&
          (data_v.stride(0) == 1) &&
          impl::cbe::is_dma_addr_ok(data_dst.ptr()) &&
          impl::cbe::is_dma_addr_ok(data_v.ptr()) &&
          impl::cbe::is_dma_addr_ok(data_m.ptr()) &&
          // must be dense to group multiple lines
          is_input_dense && is_output_dense &&
          // ensure enough lines for a group
          (mult <= data_m.size(1 - axis)) &&
          impl::cbe::Task_manager::instance()->num_spes() > 0;
      }
    }  // end if (is_row_vmmul)
    else
    {
      return 
	// (large sizes are broken down)
	(size >= VSIP_IMPL_MIN_VMMUL_SIZE) && 
	(data_dst.stride(axis) == 1) &&
	(data_m.stride(axis)   == 1) &&
	(data_v.stride(0) == 1) &&
	impl::cbe::is_dma_addr_ok(data_dst.ptr()) &&
	// (V doesn't need to be DMA aligned)
	impl::cbe::is_dma_addr_ok(data_m.ptr()) &&
	impl::cbe::is_dma_stride_ok<lhs_element_type>(data_dst.stride(1 - axis)) &&
	impl::cbe::is_dma_stride_ok<m_element_type>(data_m.stride(1 - axis)) &&
	impl::cbe::is_dma_size_ok(size * sizeof(v_value_type)) &&
	impl::cbe::Task_manager::instance()->num_spes() > 0;
    }
  }
  
  static void exec(LHS &lhs, RHS const &rhs)
  {
    VBlock const& vblock = rhs.get_vblk();
    MBlock const& mblock = rhs.get_mblk();

    Matrix<lhs_value_type, LHS> m_dst(lhs);
    const_Vector<lhs_value_type, VBlock>  v(const_cast<VBlock&>(vblock));
    const_Matrix<lhs_value_type, MBlock>  m(const_cast<MBlock&>(mblock));

    dda::Data<LHS, dda::out, lhs_lp> data_dst(lhs);
    dda::Data<VBlock, dda::in, vblock_lp> data_v(vblock);
    dda::Data<MBlock, dda::in, mblock_lp> data_m(mblock);

    typedef typename dda::Data<LHS, dda::out>::element_type lhs_element_type;
    typedef typename dda::Data<MBlock, dda::in>::element_type m_element_type;

    dimension_type const axis = (D == row ? is_row_vmmul : !is_row_vmmul);
    length_type const size = lhs.size(2, axis);
    length_type const lines = lhs.size(2, 1 - axis);

    // If we passed the rt_valid() check and the strides are not of a size
    // that can be DMA'd, then multiple rows can be grouped together.
    // This value must be recomputed though, as this is a static object
    // and cannot store the result computed previously in rt_valid().  
    // It also helps performance to skip this computation when the 
    // matrix can be processed row-by-row (e.g. stride checks pass).
    length_type mult = 1;
    if (!impl::cbe::is_dma_stride_ok<lhs_element_type>(data_dst.stride(1 - axis)) ||
        !impl::cbe::is_dma_stride_ok<m_element_type>(data_m.stride(1 - axis)))
    {
      mult = max_vectors_per_group(lines, size);
    }


    // The ct_valid check above ensures that the order taken 
    // matches the storage order if reaches this point.
    if (D == row && is_same<lhs_order_type, row2_type>::value)
    {
      impl::cbe::vmmul_row(data_v.ptr(),
			   data_m.ptr(),
			   data_dst.ptr(),
			   data_m.stride(0),   // elements between rows of source matrix
			   data_dst.stride(0), // elements between rows of destination matrix
			   lhs.size(2, 0),    // number of rows
                           lhs.size(2, 1),    // length of each row
                           mult);             // number of rows per group within a DMA transfer
    }
    else if (D == col && is_same<lhs_order_type, row2_type>::value)
    {
      impl::cbe::vmmul_col(data_v.ptr(),
			   data_m.ptr(),
			   data_dst.ptr(),
			   data_m.stride(0),   // elements between rows of source matrix
			   data_dst.stride(0), // elements between rows of destination matrix
			   lhs.size(2, 0),    // number of rows
			   lhs.size(2, 1));   // length of each row
    }
    else if (D == col && is_same<lhs_order_type, col2_type>::value)
    {
      impl::cbe::vmmul_row(data_v.ptr(),
			   data_m.ptr(),
			   data_dst.ptr(),
			   data_m.stride(1),   // elements between cols of source matrix
			   data_dst.stride(1), // elements between cols of destination matrix
			   lhs.size(2, 1),    // number of cols
                           lhs.size(2, 0),    // length of each col
                           mult);             // number of rows per group within a DMA transfer
    }
    else // if (D == row && is_same<order_type, col2_type>::value)
    {
      impl::cbe::vmmul_col(data_v.ptr(),
			   data_m.ptr(),
			   data_dst.ptr(),
			   data_m.stride(1),   // elements between cols of source matrix
			   data_dst.stride(1), // elements between cols of destination matrix
			   lhs.size(2, 1),    // number of cols
			   lhs.size(2, 0));   // length of each col
    }
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
