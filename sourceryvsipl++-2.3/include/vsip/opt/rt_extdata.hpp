/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/rt_extdata.hpp
    @author  Jules Bergmann
    @date    2005-05-03
    @brief   VSIPL++ Library: Runtime Direct Data Access.

*/

#ifndef VSIP_OPT_RT_EXTDATA_HPP
#define VSIP_OPT_RT_EXTDATA_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/extdata.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

template <typename ComplexType>
inline rt_complex_type
complex_format()
{
  if (Type_equal<ComplexType, Cmplx_inter_fmt>::value)
    return cmplx_inter_fmt;
  else
    return cmplx_split_fmt;
}


template <dimension_type D,
	  typename       Order,
	  typename       PackType,
	  typename	 ComplexType>
inline rt_complex_type
complex_format(Layout<D, Order, PackType, ComplexType>)
{
  return complex_format<ComplexType>();
}



template <dimension_type D>
inline rt_complex_type
complex_format(Rt_layout<D> const& rtl)
{
  return rtl.complex;
}



template <dimension_type D>
inline rt_complex_type
complex_format(Applied_layout<Rt_layout<D> > const& appl)
{
  return appl.complex_format();
}



template <typename PackType>
inline rt_pack_type
pack_format()
{
  if      (Type_equal<PackType, Stride_unknown>::value)
    return stride_unknown;
  else if (Type_equal<PackType, Stride_unit>::value)
    return stride_unit;
  else if (Type_equal<PackType, Stride_unit_dense>::value)
    return stride_unit_dense;
  else /* if (Type_equal<PackType, Stride_unit_align>::value) */
    return stride_unit_align;
}

template <dimension_type D,
	  typename       Order,
	  typename       PackType,
	  typename	 ComplexType>
inline rt_pack_type
pack_format(Layout<D, Order, PackType, ComplexType>)
{
  return pack_format<PackType>();
}



template <dimension_type D>
inline rt_pack_type
pack_format(Rt_layout<D> const& rtl)
{
  return rtl.pack;
}



template <dimension_type D,
	  typename       Order,
	  typename       PackType,
	  typename	 ComplexType>
inline unsigned
layout_alignment(Layout<D, Order, PackType, ComplexType>)
{
  return Is_stride_unit_align<PackType>::align;
}



template <dimension_type D>
inline unsigned
layout_alignment(Rt_layout<D> const& rtl)
{
  return rtl.align;
}


template <dimension_type D,
	  typename       OrderType,
	  typename       PackType,
	  typename	 ComplexType>
inline dimension_type
layout_nth_dim(
  Layout<D, OrderType, PackType, ComplexType> const&,
  dimension_type const d)
{
  if      (d == 0) return OrderType::impl_dim0;
  else if (d == 1) return OrderType::impl_dim1;
  else /*if (d == 2)*/ return OrderType::impl_dim2;
}



template <dimension_type D>
inline dimension_type
layout_nth_dim(Rt_layout<D> const& rtl, dimension_type const d)
{
  if      (d == 0) return rtl.order.impl_dim0;
  else if (d == 1) return rtl.order.impl_dim1;
  else /*if (d == 2)*/ return rtl.order.impl_dim2;
}



template <dimension_type D,
	  typename       Block>
inline Rt_layout<D>
block_layout(Block const&)
{
  typedef typename Block_layout<Block>::access_type  access_type;
  typedef typename Block_layout<Block>::order_type   order_type;
  typedef typename Block_layout<Block>::pack_type    pack_type;
  typedef typename Block_layout<Block>::complex_type complex_type;

  return Rt_layout<D>(
    pack_format<pack_type>(),
    Rt_tuple(order_type()),
    complex_format<complex_type>(),
    Is_stride_unit_align<pack_type>::align);
}



template <typename Block1T,
	  typename Block2T,
	  bool Is_direct = 
	     Type_equal<typename Block_layout<Block1T>::access_type,
			Direct_access_tag>::value &&
	     Type_equal<typename Block_layout<Block2T>::access_type,
			Direct_access_tag>::value,
	  bool Is_split1 = Is_split_block<Block1T>::value,
	  bool Is_split2 = Is_split_block<Block2T>::value>
struct Is_alias_helper
{
  static bool value(Block1T const&, Block2T const&) { return false; }
};

template <typename Block1T,
	  typename Block2T>
struct Is_alias_helper<Block1T, Block2T, true, false, false>
{
  static bool value(Block1T const& blk1, Block2T const& blk2)
  { return blk1.impl_data() == blk2.impl_data(); }
};

template <typename Block1T,
	  typename Block2T>
struct Is_alias_helper<Block1T, Block2T, true, true, true>
{
  static bool value(Block1T const& blk1, Block2T const& blk2)
  {
    return blk1.impl_data().first == blk2.impl_data().first &&
           blk1.impl_data().second == blk2.impl_data().second;
  }
};



/// Check if two blocks may potentially alias each other when using
/// Ext_data.

template <typename Block1T,
	  typename Block2T>
inline bool
is_alias(
  Block1T const& blk1,
  Block2T const& blk2)
{
  return Is_alias_helper<Block1T, Block2T>::value(blk1, blk2);
}



namespace data_access
{

/// Determine if direct access is OK at runtime for a given block.

template <typename BlockT,
	  typename LayoutT>
bool
is_direct_ok(
  BlockT const&  block,
  LayoutT const& layout)
{
  typedef typename BlockT::value_type                value_type;
  typedef typename Block_layout<BlockT>::layout_type block_layout_type;

  dimension_type const dim = LayoutT::dim;

  block_layout_type block_layout;

  dimension_type const dim0 = layout_nth_dim(block_layout, 0);
  dimension_type const dim1 = layout_nth_dim(block_layout, 1);
  dimension_type const dim2 = layout_nth_dim(block_layout, 2);

  if (Is_complex<value_type>::value &&
      complex_format(block_layout) != complex_format(layout))
    return false;

  for (dimension_type d=0; d<dim; ++d)
    if (layout_nth_dim(block_layout, d) != layout_nth_dim(layout, d))
      return false;

  if (pack_format(layout) == stride_unit)
  {
    if (dim == 1)
      return block.impl_stride(1, 0) == 1;
    else if (dim == 2)
      return block.impl_stride(2, dim1) == 1;
    else /* if (dim == 3) */
      return block.impl_stride(3, dim2) == 1;
  }
  else if (pack_format(layout) == stride_unit_dense)
  {
    if (dim == 1)
      return block.impl_stride(1, 0) == 1;
    else if (dim == 2)
      return    block.impl_stride(2, dim1) == 1
	     && (   block.impl_stride(2, dim0) ==
		       static_cast<stride_type>(block.size(2, dim1))
		 || block.size(2, dim0) == 1);
    else /* if (dim == 3) */
    {

      bool ok2 = (block.impl_stride(3, dim2) == 1);
      bool ok1 = (block.impl_stride(3, dim1) ==
		  static_cast<stride_type>(block.size(3, dim2)))
	|| (block.size(3, dim1) == 1 && block.size(3, dim0) == 1);
      bool ok0 = (block.impl_stride(3, dim0) ==
		  static_cast<stride_type>(block.size(3, dim1) *
					   block.size(3, dim2)))
	|| block.size(3, dim0) == 1;

      return ok0 && ok1 && ok2;
    }
  }
  else if (pack_format(layout) == stride_unit_align)
  {
    // unsigned align = Is_stride_unit_align<typename LP::pack_type>::align;
    unsigned align = layout_alignment(layout);

    if (!data_access::is_aligned_to(block.impl_data(), align))
      return false;

    if (dim == 1)
      return block.impl_stride(1, 0) == 1;
    else if (dim == 2)
      return block.impl_stride(2, dim1) == 1 &&
	((block.impl_stride(2, dim0) * sizeof(value_type)) % align == 0);
    else /* if (LP::dim == 3) */
      return 
	block.impl_stride(3, dim2) == 1 &&
	(block.impl_stride(3, dim1) * sizeof(value_type)) % align == 0 &&
	(block.impl_stride(3, dim0) * sizeof(value_type)) % align == 0;
  }
  else /* if (Type_equal<typename LP::pack_type, Stride_unknown>::value) */
  {
    assert(pack_format(layout) == stride_unknown);
    return true;
  }
}



/// Run-time low-level data access class.

/// Requires
///   AT to be an access-type tag,
///   BLOCK to be a VSIPL++ block type,
///   DIM to be the dimension of the desired run-time layout.

template <typename       AT,
	  typename       Block,
	  dimension_type Dim>
class Rt_low_level_data_access;



/// Specialization for low-level direct data access.

/// Depending on the requested run-time layout, data will either be
/// accessed directly, or will be copied into a temporary buffer.
///
/// Requires:
///   BLOCK to be a block that supports direct access via member
///     functions impl_data() and impl_stride().  Access to these
///     members can be protected by making Low_level_data_access a friend
///     class to the block.
///   DIM to be the dimension of the desired run-time layout.

template <typename       Block,
	  dimension_type Dim>
class Rt_low_level_data_access<Direct_access_tag, Block, Dim>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = Dim;

  typedef typename Block::value_type value_type;
  typedef Rt_allocated_storage<value_type> storage_type;
  typedef Rt_pointer<value_type> raw_ptr_type;

  // Compile- and run-time properties.
public:
  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int    cost         (Block const& block, Rt_layout<Dim> const& rtl)
    { return is_direct_ok(block, rtl) ? 0 : 2; }

  static size_t mem_required (Block const& block, Rt_layout<Dim> const& )
    { return sizeof(typename Block::value_type) * block.size(); }
  static size_t xfer_required(Block const& , Rt_layout<Dim> const& )
    { return !CT_Xfer_not_req; }

  // Constructor and destructor.
public:
  Rt_low_level_data_access(
    Block&                blk,
    Rt_layout<Dim> const& rtl,
    bool                  no_preserve,
    raw_ptr_type          buffer = NULL,
    length_type           buffer_size = 0)
  : use_direct_(!no_preserve && is_direct_ok(blk, rtl)),
    app_layout_(use_direct_ ?
		Applied_layout<Rt_layout<Dim> >(empty_layout) :
		Applied_layout<Rt_layout<Dim> >(
		  rtl, extent<dim>(blk), sizeof(value_type))),
    storage_(use_direct_ ? 0 : app_layout_.total_size(), rtl.complex, buffer)
  {
    assert(use_direct_ ||
	   buffer_size == 0 || app_layout_.total_size() <= buffer_size);
  }

  ~Rt_low_level_data_access()
    { if (storage_.is_alloc()) storage_.deallocate(app_layout_.total_size()); }

  void begin(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Rt_block_copy_to_ptr<Dim, Block>::copy(blk, app_layout_,
					     storage_.data());
  }

  void end(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Rt_block_copy_from_ptr<Dim, Block>::copy(blk, app_layout_,
					       storage_.data());
  }

  int cost() const { return use_direct_ ? 0 : 2; }

  // Direct data acessors.
public:
  raw_ptr_type data(Block* blk) const
    { return use_direct_ ? raw_ptr_type(blk->impl_data()) : storage_.data(); }

  stride_type stride(Block* blk, dimension_type d) const
    { return use_direct_ ? blk->impl_stride(dim, d) : app_layout_.stride(d);  }

  length_type size(Block* blk, dimension_type d) const
    { return use_direct_ ? blk->size(dim, d) : blk->size(Block::dim, d); }

  // Member data.
private:
  bool                            use_direct_;
  Applied_layout<Rt_layout<Dim> > app_layout_;
  storage_type                    storage_;
};



/// Specialization for low-level copied data access.

/// Requires:
///   BLOCK to be a block.
///   DIM to be the dimension of the desired run-time layout.

template <typename       Block,
	  dimension_type Dim>
class Rt_low_level_data_access<Copy_access_tag, Block, Dim>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = Dim;

  typedef typename Block::value_type value_type;
  typedef Rt_allocated_storage<value_type> storage_type;
  typedef Rt_pointer<value_type> raw_ptr_type;

  // Compile- and run-time properties.
public:
  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int    cost         (Block const& block, Rt_layout<Dim> const& rtl)
    { return is_direct_ok(block, rtl) ? 0 : 2; }

  static size_t mem_required (Block const& block, Rt_layout<Dim> const& )
    { return sizeof(typename Block::value_type) * block.size(); }
  static size_t xfer_required(Block const& , Rt_layout<Dim> const& )
    { return !CT_Xfer_not_req; }

  // Constructor and destructor.
public:
  Rt_low_level_data_access(
    Block&                blk,
    Rt_layout<Dim> const& rtl,
    bool                  /*no_preserve*/,
    raw_ptr_type          buffer = NULL,
    length_type           buffer_size = 0)
  : app_layout_(Applied_layout<Rt_layout<Dim> >(
		  rtl, extent<dim>(blk), sizeof(value_type))),
    storage_(app_layout_.total_size(), rtl.complex, buffer)
  {
    assert(buffer_size == 0 || app_layout_.total_size() <= buffer_size);
  }

  ~Rt_low_level_data_access()
    { if (storage_.is_alloc()) storage_.deallocate(app_layout_.total_size()); }

  void begin(Block* blk, bool sync)
  {
    if (sync)
    {
      Rt_block_copy_to_ptr<Dim, Block>::copy(blk, app_layout_,
					     storage_.data());
    }
  }

  void end(Block* blk, bool sync)
  {
    if (sync)
      Rt_block_copy_from_ptr<Dim, Block>::copy(blk, app_layout_,
					       storage_.data());
  }

  int cost() const { return 2; }

  // Direct data acessors.
public:
  raw_ptr_type data(Block*) const
    { return storage_.data(); }

  stride_type stride(Block*, dimension_type d) const
    { return app_layout_.stride(d);  }

  length_type size(Block* blk, dimension_type d) const
    { return blk->size(Block::dim, d); }

  // Member data.
private:
  Applied_layout<Rt_layout<Dim> > app_layout_;
  storage_type                    storage_;
};

} // namespace vsip::impl::data_access



/// Run-time high-level data access class.  Provides data access to data
/// stored in blocks, using an appropriate low-level data interface.
///
/// Template parameters:
///   :Block: is a block type.  It should not be const.
///   :Dim: is the dimension of the run-time layout policy for data access.
///   :RP: is a reference counting policy.
///
/// Notes:
///  - When using Rt_ext_data to access a const block:
///     - :Block: should be the non-const base block type.
///     - Rt_ext_data should be constructed with a const block reference.
template <typename       Block,
	  dimension_type Dim = Block_layout<Block>::dim,
	  typename       RP  = No_count_policy>
class Rt_ext_data : Assert_proper_block<Block>
{
  // Compile time typedefs.
public:
  typedef typename Non_const_of<Block>::type non_const_block_type;

  typedef typename Block_layout<Block>::access_type              AT;
  typedef data_access::Rt_low_level_data_access<AT, Block, Dim>  ext_type;
  typedef typename Block::value_type                             value_type;
  typedef Rt_pointer<value_type>                                 raw_ptr_type;


  // Compile- and run-time properties.
public:
  static int   const CT_Cost          = ext_type::CT_Cost;
  static bool  const CT_Mem_not_req   = ext_type::CT_Mem_not_req;
  static bool  const CT_Xfer_not_req  = ext_type::CT_Xfer_not_req;


  // Constructor and destructor.
public:
  Rt_ext_data(non_const_block_type& block,
	      Rt_layout<Dim> const& rtl,
	      sync_action_type      sync   = SYNC_INOUT,
	      raw_ptr_type          buffer = 0, /*storage_type::null()*/
	      length_type           buffer_size = 0)
    : blk_    (&block),
      ext_    (block, rtl, sync & SYNC_NOPRESERVE_impl, buffer, buffer_size),
      sync_   (sync)
    { ext_.begin(blk_.get(), sync_ & SYNC_IN); }

  Rt_ext_data(Block const&          block,
	      Rt_layout<Dim> const& rtl,
	      sync_action_type      sync,
	      raw_ptr_type          buffer = 0, /*storage_type::null()*/
	      length_type           buffer_size = 0)
    : blk_ (&const_cast<Block&>(block)),
      ext_ (const_cast<Block&>(block), rtl,
	    sync & SYNC_NOPRESERVE_impl, buffer, buffer_size),
      sync_(sync)
  {
    assert(sync != SYNC_OUT && sync != SYNC_INOUT);
    ext_.begin(blk_.get(), sync_ & SYNC_IN);
  }

  ~Rt_ext_data()
    { ext_.end(blk_.get(), sync_ & SYNC_OUT); }

  // Direct data acessors.
public:
  raw_ptr_type data() const
    { return ext_.data(blk_.get()); }

  stride_type  stride(dimension_type d) const
    { return ext_.stride(blk_.get(), d); }

  length_type  size(dimension_type d) const
    { return ext_.size(blk_.get(), d); }

  int           cost  () const
    { return ext_.cost(); }

  // Member data.
private:
  typename View_block_storage<Block>::template With_rp<RP>::type
		   blk_;
  ext_type         ext_;
  sync_action_type sync_;
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_RT_EXTDATA_HPP
