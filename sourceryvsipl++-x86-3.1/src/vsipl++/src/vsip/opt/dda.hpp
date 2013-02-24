/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */
#ifndef vsip_opt_dda_hpp_
#define vsip_opt_dda_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/static_assert.hpp>
#include <vsip/core/block_copy.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/opt/choose_access.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/dda.hpp>

namespace vsip
{
namespace dda
{
namespace impl
{

/// Specialization for low-level reordered direct data access.
/// (Not implemented yet).

/// Template parameters:
///   :Block: to be a block that supports direct access via member
///           functions ptr() and stride().  Access to these
///           members can be protected by making Accessor a friend
///           class to the block.
///   :LP:    is a layout policy describing the desired layout.  It is should
///           match the inherent layout of the block.  Specifying a layout
///           not directly supported by the block is an error and results in
///           undefined behavior.
template <typename Block,
	  typename LP>
class Accessor<Block, LP, Reorder_access_tag>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  static pack_type const packing = LP::packing;
  static storage_format_type const storage_format = LP::storage_format;

  typedef Storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;
  // typedef typename storage_type::type raw_ptr_type;

  // Compile- and run-time properties.
public:
  static int   const CT_Cost         = 1;
  static bool  const CT_Mem_not_req  = true;
  static bool  const CT_Xfer_not_req = false;

  static int    cost         (Block const& /*block*/, LP const& /*layout*/)
    { return CT_Cost; }
  static size_t mem_required (Block const& /*block*/, LP const& /*layout*/)
    { return 0; }
  static size_t xfer_required(Block const& /*block*/, LP const& /*layout*/)
    { return !CT_Xfer_not_req; }

  // Constructor and destructor.
public:
  Accessor(Block&, non_const_ptr_type = non_const_ptr_type())
  {
    VSIP_IMPL_THROW(impl::unimplemented("Reorder_access_tag"));
  }

  ~Accessor() {}

  void cleanup(Block*)
  {
    VSIP_IMPL_THROW(impl::unimplemented("Reorder_access_tag"));
  }

  // Direct data acessors.
public:
  value_type *ptr(Block* blk) const { return blk->ptr();}
  stride_type stride(Block* blk, dimension_type d) const { return blk->stride(dim, d);}
  length_type size(Block* blk, dimension_type d) const { return blk->size(dim, d);}
  length_type size(Block* blk) const { return blk->size();}
};



template <typename T>
inline bool
is_aligned_to(T* pointer, size_t align)
{
  return reinterpret_cast<size_t>(pointer) % align == 0;
}

template <typename T>
inline bool
is_aligned_to(std::pair<T*, T*> pointer, size_t align)
{
  return reinterpret_cast<size_t>(pointer.first)  % align == 0 &&
         reinterpret_cast<size_t>(pointer.second) % align == 0;
}


/// Determine if direct access is OK at runtime for a given block.
template <typename LP,
	  typename BlockT>
bool
is_direct_ok(BlockT const& block)
{
  typedef typename BlockT::value_type                 value_type;
  static storage_format_type const storage_format = get_block_layout<BlockT>::storage_format;
  typedef typename get_block_layout<BlockT>::order_type   order_type;

  dimension_type const dim0 = order_type::impl_dim0;
  dimension_type const dim1 = order_type::impl_dim1;
  dimension_type const dim2 = order_type::impl_dim2;

  if (storage_format != LP::storage_format ||
      !is_same<order_type, typename LP::order_type>::value)
    return false;
  else if (LP::packing == unit_stride)
  {
    if (LP::dim == 1)
      return block.stride(1, 0) == 1;
    else if (LP::dim == 2)
      return block.stride(2, dim1) == 1;
    else /* if (LP::dim == 3) */
      return block.stride(3, dim2) == 1;
  }
  else if (LP::packing == dense)
  {
    if (LP::dim == 1)
      return block.stride(1, 0) == 1;
    else if (LP::dim == 2)
      return block.stride(2, dim1) == 1
	     && (block.stride(2, dim0) ==
		 static_cast<stride_type>(block.size(2, dim1))
		 || block.size(2, dim0) == 1);
    else /* if (LP::dim == 3) */
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
  else if (is_packing_aligned<LP::packing>::value)
  {
    unsigned alignment = is_packing_aligned<LP::packing>::alignment;

    if (!is_aligned_to(block.ptr(), alignment)) return false;

    if (LP::dim == 1)
      return block.stride(1, 0) == 1;
    else if (LP::dim == 2)
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
    assert(LP::packing == any_packing);
    return true;
  }
}



/// Specialization for flexible (direct or copied, depending on
/// stride) direct data access.
///
/// Template parameter:
///   :Block: to be a block.
///   :LP:    is a layout policy describing the desired layout.
///           The desired layout can be different from the block's layout.
///
/// Notes:
///   When the desired layout packing format is either packing::unit_stride or
///   packing::unknown, the packing format used will be packing::dense.
template <typename Block,
	  typename LP>
class Accessor<Block, LP, Flexible_access_tag>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  static pack_type const packing = 
    LP::packing == unit_stride || LP::packing == any_packing ? dense : LP::packing;
  static storage_format_type const storage_format = LP::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> actual_layout_type;

  typedef Allocated_storage<storage_format, value_type> storage_type;
  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;
  // typedef typename storage_type::type                 raw_ptr_type;
  // typedef typename storage_type::const_type           const_raw_ptr_type;

  // Compile- and run-time properties.
public:
  static int   const CT_Cost          = 2;
  static bool  const CT_Mem_not_req   = false;
  static bool  const CT_Xfer_not_req  = false;

  static int    cost         (Block const& block, LP const&)
    { return is_direct_ok<LP>(block) ? 0 : 2; }
  static size_t mem_required (Block const& block, LP const&)
    { return is_direct_ok<LP>(block) ? 0 :
	sizeof(typename Block::value_type) * block.size(); }
  static size_t xfer_required(Block const& block, LP const&)
    { return is_direct_ok<LP>(block) ? false : !CT_Xfer_not_req; }

  // Constructor and destructor.
public:
  Accessor(Block &blk, non_const_ptr_type buffer = non_const_ptr_type())
    : use_direct_(is_direct_ok<LP>(blk)),
      layout_    (extent<dim>(blk)),
      storage_   (use_direct_ ? 0 : layout_.total_size(), buffer)
  {}

  ~Accessor() { storage_.deallocate(layout_.total_size());}

  void begin(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Block_copy_to_ptr<LP::dim, Block, order_type, packing, storage_format>
	::copy(blk, layout_, storage_.ptr());
  }

  void end(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Block_copy_from_ptr<LP::dim, Block, order_type, packing, storage_format>
	::copy(blk, layout_, storage_.ptr());
  }

  int cost() const { return use_direct_ ? 0 : 2; }

  // Direct data acessors.
public:
  ptr_type ptr(Block* blk)
    { return use_direct_ ? blk->ptr() : storage_.ptr(); }
  const_ptr_type ptr(Block* blk) const
    { return use_direct_ ? blk->ptr() : storage_.ptr(); }
  stride_type	stride(Block* blk, dimension_type d) const
    { return use_direct_ ? blk->stride(dim, d) : layout_.stride(d); }
  length_type	size  (Block* blk, dimension_type d) const
    { return use_direct_ ? blk->size(dim, d) : blk->size(Block::dim, d); }
  length_type	size  (Block* blk) const
    { return blk->size(); }

  // Member data.
private:
  bool                               use_direct_;
  Applied_layout<actual_layout_type> layout_;
  storage_type                       storage_;
};

template <> struct Cost<Reorder_access_tag>  { static int const value = 1; };
template <> struct Cost<Flexible_access_tag> { static int const value = 2; };

/// Class describing access type demotion.
template <typename AccessTag>
struct Access_demotion;

template <>
struct Access_demotion<Direct_access_tag>
{
  typedef Direct_access_tag  direct_type;

  typedef Copy_access_tag     reorder_type;
  typedef Flexible_access_tag flex_type;
  typedef Copy_access_tag     copy_type;
};

template <>
struct Access_demotion<Reorder_access_tag>
{
  typedef Reorder_access_tag direct_type;
  typedef Reorder_access_tag reorder_type;
  typedef Copy_access_tag    flex_type;
  typedef Copy_access_tag    copy_type;
};

template <>
struct Access_demotion<Copy_access_tag>
{
  typedef Copy_access_tag    direct_type;
  typedef Copy_access_tag    reorder_type;
  typedef Copy_access_tag    flex_type;
  typedef Copy_access_tag    copy_type;
};



/// Choose access type for a given block and desired layout.
template <typename Block,
	  typename LP>
struct Choose_access
{
  typedef typename vsip::impl::remove_const<Block>::type block_type;
  typedef typename conditional<supports_dda<block_type>::value,
			       Direct_access_tag, Copy_access_tag>::type access_type;
  typedef Access_demotion<access_type>              demotion_type;

  typedef
  CA_General<demotion_type,
	     typename get_block_layout<block_type>::order_type,
	     get_block_layout<block_type>::packing,
	     get_block_layout<block_type>::storage_format,
             typename LP::order_type,
             LP::packing,
	     LP::storage_format> ca_type;

  typedef typename ca_type::type type;
  typedef typename ca_type::reason_type reason_type;
};

} // namespace vsip::dda::impl
} // namespace vsip::dda
} // namespace vsip

#endif
