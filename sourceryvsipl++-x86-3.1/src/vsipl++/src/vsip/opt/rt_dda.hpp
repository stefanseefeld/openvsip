/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */
#ifndef vsip_opt_rt_dda_hpp_
#define vsip_opt_rt_dda_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/dda.hpp>

namespace vsip
{
namespace impl
{

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline storage_format_type 
layout_storage_format(Layout<D, O, P, C>)
{ return C;}

template <dimension_type D>
inline storage_format_type 
layout_storage_format(Rt_layout<D> const &rtl) 
{ return rtl.storage_format;}

template <dimension_type D>
inline storage_format_type 
layout_storage_format(Applied_layout<Rt_layout<D> > const& appl)
{ return appl.storage_format();}

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline pack_type 
layout_packing(Layout<D, O, P, C>)
{ return P;}

template <dimension_type D>
inline pack_type 
layout_packing(Rt_layout<D> const &rtl)
{ return rtl.packing;}

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline unsigned 
layout_alignment(Layout<D, O, P, C>)
{ return is_packing_aligned<P>::alignment;}

template <dimension_type D>
inline unsigned 
layout_alignment(Rt_layout<D> const &rtl)
{ return rtl.alignment;}

template <dimension_type D, typename O, pack_type P, storage_format_type C>
inline dimension_type 
layout_nth_dim(Layout<D, O, P, C> const&, dimension_type const d)
{
  if      (d == 0) return O::impl_dim0;
  else if (d == 1) return O::impl_dim1;
  else /*if (d == 2)*/ return O::impl_dim2;
}

template <dimension_type D>
inline dimension_type 
layout_nth_dim(Rt_layout<D> const& rtl, dimension_type const d)
{
  if      (d == 0) return rtl.order.impl_dim0;
  else if (d == 1) return rtl.order.impl_dim1;
  else /*if (d == 2)*/ return rtl.order.impl_dim2;
}

template <dimension_type D, typename Block>
inline Rt_layout<D>
block_layout(Block const&)
{
  return Rt_layout<D>(get_block_layout<Block>::packing,
		      Rt_tuple(typename get_block_layout<Block>::order_type()),
		      get_block_layout<Block>::storage_format,
		      is_packing_aligned<get_block_layout<Block>::packing>::alignment);
}

template <typename Block1T,
	  typename Block2T,
	  bool Is_direct = 
	     supports_dda<Block1T>::value &&
	     supports_dda<Block2T>::value,
	  bool Is_split1 = is_split_block<Block1T>::value,
	  bool Is_split2 = is_split_block<Block2T>::value>
struct Is_alias_helper
{
  static bool value(Block1T const&, Block2T const&) { return false; }
};

template <typename Block1T,
	  typename Block2T>
struct Is_alias_helper<Block1T, Block2T, true, false, false>
{
  static bool value(Block1T const& blk1, Block2T const& blk2)
  { return blk1.ptr() == blk2.ptr();}
};

template <typename Block1T,
	  typename Block2T>
struct Is_alias_helper<Block1T, Block2T, true, true, true>
{
  static bool value(Block1T const& blk1, Block2T const& blk2)
  {
    return blk1.ptr().first == blk2.ptr().first &&
           blk1.ptr().second == blk2.ptr().second;
  }
};

/// Check if two blocks may potentially alias each other when using
/// dda::Data.
template <typename Block1T,
	  typename Block2T>
inline bool
is_alias(Block1T const& blk1, Block2T const& blk2)
{
  return Is_alias_helper<Block1T, Block2T>::value(blk1, blk2);
}

} // namespace vsip::impl

namespace dda
{
namespace impl
{
/// Determine if direct access is OK at runtime for a given block.

template <typename BlockT,
	  typename LayoutT>
bool
is_direct_ok(BlockT const&  block, LayoutT const& layout)
{
  typedef typename BlockT::value_type                value_type;
  typedef typename get_block_layout<BlockT>::type block_layout_type;

  dimension_type const dim = LayoutT::dim;

  block_layout_type block_layout;

  dimension_type const dim0 = layout_nth_dim(block_layout, 0);
  dimension_type const dim1 = layout_nth_dim(block_layout, 1);
  dimension_type const dim2 = layout_nth_dim(block_layout, 2);

  if (is_complex<value_type>::value &&
      layout_storage_format(block_layout) != layout_storage_format(layout))
    return false;

  for (dimension_type d=0; d<dim; ++d)
    if (layout_nth_dim(block_layout, d) != layout_nth_dim(layout, d))
      return false;

  if (layout_packing(layout) == unit_stride)
  {
    if (dim == 1) return block.stride(1, 0) == 1;
    else if (dim == 2) return block.stride(2, dim1) == 1;
    else return block.stride(3, dim2) == 1;
  }
  else if (layout_packing(layout) == dense)
  {
    if (dim == 1) return block.stride(1, 0) == 1;
    else if (dim == 2)
      return block.stride(2, dim1) == 1
	&& (block.stride(2, dim0) == static_cast<stride_type>(block.size(2, dim1))
	    || block.size(2, dim0) == 1);
    else /* if (dim == 3) */
    {
      bool ok2 = (block.stride(3, dim2) == 1);
      bool ok1 = (block.stride(3, dim1) ==
		  static_cast<stride_type>(block.size(3, dim2)))
	|| (block.size(3, dim1) == 1 && block.size(3, dim0) == 1);
      bool ok0 = (block.stride(3, dim0) ==
		  static_cast<stride_type>(block.size(3, dim1) *
					   block.size(3, dim2)))
	|| block.size(3, dim0) == 1;

      return ok0 && ok1 && ok2;
    }
  }
  else if (is_aligned(layout_packing(layout)))
  {
    // unsigned align = Is_stride_unit_align<typename LP::packing>::align;
    unsigned alignment = layout_alignment(layout);

    if (!is_aligned_to(block.ptr(), alignment))
      return false;

    if (dim == 1)
      return block.stride(1, 0) == 1;
    else if (dim == 2)
      return block.stride(2, dim1) == 1 &&
	((block.stride(2, dim0) * sizeof(value_type)) % alignment == 0);
    else /* if (LP::dim == 3) */
      return 
	block.stride(3, dim2) == 1 &&
	(block.stride(3, dim1) * sizeof(value_type)) % alignment == 0 &&
	(block.stride(3, dim0) * sizeof(value_type)) % alignment == 0;
  }
  else /* if (is_same<typename LP::packing, packing::unknown>::value) */
  {
    assert(layout_packing(layout) == any_packing);
    return true;
  }
}



/// Run-time low-level data access class.
///
/// Template parameters:
///   :Block: the Block type.
///   :D: the dimension of the desired run-time layout.
///   :AT: the access-type tag,
template <typename Block, dimension_type D, typename AT>
class Rt_Accessor;

/// Depending on the requested run-time layout, data will either be
/// accessed directly, or will be copied into a temporary buffer.
///
/// Template parameters:
///   :Block: the block type, supporting direct access via member
///           functions ptr() and stride().
///   :D: the dimension of the desired run-time layout.
template <typename Block, dimension_type D>
class Rt_Accessor<Block, D, Direct_access_tag>
{
public:
  static dimension_type const dim = D;

  typedef typename Block::value_type value_type;
  typedef Rt_allocated_storage<value_type> storage_type;
  typedef typename dda::impl::Pointer<value_type> non_const_ptr_type;
  typedef typename dda::impl::const_Pointer<value_type> const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int cost(Block const& block, Rt_layout<D> const& rtl)
  { return is_direct_ok(block, rtl) ? 0 : 2; }

  static size_t mem_required (Block const& block, Rt_layout<D> const& )
  { return sizeof(typename Block::value_type) * block.size(); }
  static size_t xfer_required(Block const& , Rt_layout<D> const& )
  { return !CT_Xfer_not_req; }

  Rt_Accessor(Block &blk,
	      Rt_layout<D> const& rtl,
	      bool force_copy,
	      non_const_ptr_type buffer = non_const_ptr_type(),
	      length_type buffer_size = 0)
    : use_direct_(!force_copy && is_direct_ok(blk, rtl)),
      app_layout_(use_direct_ ?
		  Applied_layout<Rt_layout<D> >(empty_layout) :
		  Applied_layout<Rt_layout<D> >(rtl, extent<dim>(blk), sizeof(value_type))),
      storage_(use_direct_ ? 0 : app_layout_.total_size(), rtl.storage_format, buffer)
  {
    assert(use_direct_ || buffer_size == 0 || app_layout_.total_size() <= buffer_size);
  }

  ~Rt_Accessor()
    { if (storage_.is_alloc()) storage_.deallocate(app_layout_.total_size());}

  void begin(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Rt_block_copy_to_ptr<D, Block>::copy(blk, app_layout_, storage_.ptr());
  }

  void end(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Rt_block_copy_from_ptr<D, Block>::copy(blk, app_layout_, storage_.ptr());
  }

  int cost() const { return use_direct_ ? 0 : 2;}

  ptr_type ptr(Block* blk) const
  { return use_direct_ ? ptr_type(blk->ptr()) : storage_.ptr();}

  stride_type stride(Block* blk, dimension_type d) const
  { return use_direct_ ? blk->stride(dim, d) : app_layout_.stride(d);}

  length_type size(Block* blk, dimension_type d) const
  { return use_direct_ ? blk->size(dim, d) : app_layout_.size(d);}
  length_type size(Block* blk) const
  { return use_direct_ ? blk->size() : app_layout_.total_size();}

private:
  bool                            use_direct_;
  Applied_layout<Rt_layout<D> > app_layout_;
  storage_type                    storage_;
};

/// Specialization for low-level copied data access.

/// Requires:
///   BLOCK to be a block.
///   DIM to be the dimension of the desired run-time layout.
template <typename Block, dimension_type D>
class Rt_Accessor<Block, D, Copy_access_tag>
{
public:
  static dimension_type const dim = D;

  typedef typename Block::value_type value_type;
  typedef Rt_allocated_storage<value_type> storage_type;
  typedef typename dda::impl::Pointer<value_type> non_const_ptr_type;
  typedef typename dda::impl::const_Pointer<value_type> const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int    cost         (Block const& block, Rt_layout<D> const& rtl)
    { return is_direct_ok(block, rtl) ? 0 : 2; }

  static size_t mem_required (Block const& block, Rt_layout<D> const& )
    { return sizeof(typename Block::value_type) * block.size(); }
  static size_t xfer_required(Block const& , Rt_layout<D> const& )
    { return !CT_Xfer_not_req; }

  Rt_Accessor(Block &blk,
	      Rt_layout<D> const& rtl,
	      bool                  /*force_copy*/,
	      non_const_ptr_type buffer = non_const_ptr_type(),
	      length_type           buffer_size = 0)
    : app_layout_(Applied_layout<Rt_layout<D> >(rtl, extent<dim>(blk), sizeof(value_type))),
      storage_(app_layout_.total_size(), rtl.storage_format, buffer)
  {
    assert(buffer_size == 0 || app_layout_.total_size() <= buffer_size);
  }

  ~Rt_Accessor()
  { if (storage_.is_alloc()) storage_.deallocate(app_layout_.total_size());}

  void begin(Block* blk, bool sync)
  {
    if (sync)
      Rt_block_copy_to_ptr<D, Block>::copy(blk, app_layout_, storage_.ptr());
  }

  void end(Block* blk, bool sync)
  {
    if (sync)
      Rt_block_copy_from_ptr<D, Block>::copy(blk, app_layout_, storage_.ptr());
  }

  int cost() const { return 2; }

  ptr_type ptr(Block*) const { return storage_.ptr();}
  stride_type stride(Block*, dimension_type d) const { return app_layout_.stride(d);}
  length_type size(Block* blk, dimension_type d) const { return blk->size(Block::dim, d);}

private:
  Applied_layout<Rt_layout<D> > app_layout_;
  storage_type storage_;
};

} // namespace vsip::dda::impl
} // namespace vsip::dda
} // namespace vsip

#endif
