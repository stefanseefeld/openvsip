/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/extdata.hpp
    @author  Jules Bergmann
    @date    2005-02-11
    @brief   VSIPL++ Library: Direct Data Access extensions for Optimized
             Library.

    This file is included by core/extdata.hpp when appropriate.  It
    should not be included directly by other source files.
*/

#ifndef VSIP_OPT_EXTDATA_HPP
#define VSIP_OPT_EXTDATA_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/static_assert.hpp>
#include <vsip/core/extdata_common.hpp>
#include <vsip/core/block_copy.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/opt/choose_access.hpp>
#include <vsip/core/domain_utils.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Namespace for low-level data access interfaces.  These interfaces
/// provide low-level data access to data stored within blocks
/// (directly or indirectly).
///
/// These interfaces are not intended to be used in application code,
/// or in the library implementation outside of the Ext_data class.
/// Not all low-level interfaces are valid for all blocks, and over time
/// details of the low-level interface may change.  To provide a
/// consistent data interface to all blocks, the Ext_data class should
/// be used instead.
namespace data_access 
{

/// Low_level_data_access declared in extdata_common.hpp.

/// Specializaitons for
///  - Direct_access_tag
///  - Copy_access_tag
/// are defined in core/extdata.hpp.



/// Specialization for low-level reordered direct data access.
/// (Not implemented yet).

/// Template parameters:
///   :Block: to be a block that supports direct access via member
///           functions impl_data() and impl_stride().  Access to these
///           members can be protected by making Low_level_data_access a friend
///           class to the block.
///   :LP:    is a layout policy describing the desired layout.  It is should
///           match the inherent layout of the block.  Specifying a layout
///           not directly supported by the block is an error and results in
///           undefined behavior.
template <typename Block,
	  typename LP>
class Low_level_data_access<Reorder_access_tag, Block, LP>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  typedef typename LP::pack_type     pack_type;
  typedef typename LP::complex_type  complex_type;

  typedef Storage<complex_type, value_type> storage_type;
  typedef typename storage_type::type raw_ptr_type;

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
  Low_level_data_access(Block&,
		     raw_ptr_type     = NULL)
  {
    VSIP_IMPL_THROW(impl::unimplemented("Reorder_access_tag"));
  }

  ~Low_level_data_access() {}

  void cleanup(Block*)
  {
    VSIP_IMPL_THROW(impl::unimplemented("Reorder_access_tag"));
  }

  // Direct data acessors.
public:
  value_type*	data  (Block* blk) const
    { return blk->impl_data(); }
  stride_type	stride(Block* blk, dimension_type d) const
    { return blk->impl_stride(dim, d); }
  length_type	size  (Block* blk, dimension_type d) const
    { return blk->size(dim, d); }
  length_type	size  (Block* blk) const
    { return blk->size(); }
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
  typedef typename Block_layout<BlockT>::complex_type complex_type;
  typedef typename Block_layout<BlockT>::order_type   order_type;

  dimension_type const dim0 = order_type::impl_dim0;
  dimension_type const dim1 = order_type::impl_dim1;
  dimension_type const dim2 = order_type::impl_dim2;

  if (!Type_equal<complex_type, typename LP::complex_type>::value ||
      !Type_equal<order_type, typename LP::order_type>::value)
    return false;
  else if (Type_equal<typename LP::pack_type, Stride_unit>::value)
  {
    if (LP::dim == 1)
      return block.impl_stride(1, 0) == 1;
    else if (LP::dim == 2)
      return block.impl_stride(2, dim1) == 1;
    else /* if (LP::dim == 3) */
      return block.impl_stride(3, dim2) == 1;
  }
  else if (Type_equal<typename LP::pack_type, Stride_unit_dense>::value)
  {
    if (LP::dim == 1)
      return block.impl_stride(1, 0) == 1;
    else if (LP::dim == 2)
      return    block.impl_stride(2, dim1) == 1
	     && (   block.impl_stride(2, dim0) ==
		       static_cast<stride_type>(block.size(2, dim1))
		 || block.size(2, dim0) == 1);
    else /* if (LP::dim == 3) */
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
  else if (Is_stride_unit_align<typename LP::pack_type>::value)
  {
    unsigned align = Is_stride_unit_align<typename LP::pack_type>::align;

    if (!is_aligned_to(block.impl_data(), align))
      return false;

    if (LP::dim == 1)
      return block.impl_stride(1, 0) == 1;
    else if (LP::dim == 2)
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
    assert((Type_equal<typename LP::pack_type, Stride_unknown>::value));
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
///   When the desired layout packing format is either Stride_unit or
///   Stride_unknown, the packing format used will be Stride_unit_dense.
template <typename Block,
	  typename LP>
class Low_level_data_access<Flexible_access_tag, Block, LP>
{
  // Compile time typedefs.
public:
  static dimension_type const dim = LP::dim;

  typedef typename Block::value_type value_type;
  typedef typename LP::order_type    order_type;
  typedef typename
          ITE_Type<Type_equal<typename LP::pack_type, Stride_unit>::value ||
	           Type_equal<typename LP::pack_type, Stride_unknown>::value,
                   As_type<Stride_unit_dense>,
		   As_type<typename LP::pack_type> >::type pack_type;
  typedef typename LP::complex_type  complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> actual_layout_type;

  typedef Allocated_storage<complex_type, value_type> storage_type;
  typedef typename storage_type::type                 raw_ptr_type;
  typedef typename storage_type::const_type           const_raw_ptr_type;

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
  Low_level_data_access(Block&         blk,
			raw_ptr_type   buffer = NULL)
    : use_direct_(is_direct_ok<LP>(blk)),
      layout_    (extent<dim>(blk)),
      storage_   (use_direct_ ? 0 : layout_.total_size(), buffer)
  {}

  ~Low_level_data_access()
    { storage_.deallocate(layout_.total_size()); }

  void begin(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Block_copy_to_ptr<LP::dim, Block, order_type, pack_type, complex_type>
	::copy(blk, layout_, storage_.data());
  }

  void end(Block* blk, bool sync)
  {
    if (!use_direct_ && sync)
      Block_copy_from_ptr<LP::dim, Block, order_type, pack_type, complex_type>
	::copy(blk, layout_, storage_.data());
  }

  int cost() const { return use_direct_ ? 0 : 2; }

  // Direct data acessors.
public:
  raw_ptr_type	data(Block* blk)
    { return use_direct_ ? blk->impl_data() : storage_.data(); }
  const_raw_ptr_type data(Block* blk) const
    { return use_direct_ ? blk->impl_data() : storage_.data(); }
  stride_type	stride(Block* blk, dimension_type d) const
    { return use_direct_ ? blk->impl_stride(dim, d) : layout_.stride(d); }
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

} // namespace vsip::impl::data_access



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
  typedef Bogus_access_tag    bogus_type;
};

template <>
struct Access_demotion<Reorder_access_tag>
{
  typedef Reorder_access_tag direct_type;
  typedef Reorder_access_tag reorder_type;
  typedef Copy_access_tag    flex_type;
  typedef Copy_access_tag    copy_type;
  typedef Bogus_access_tag   bogus_type;
};

template <>
struct Access_demotion<Copy_access_tag>
{
  typedef Copy_access_tag    direct_type;
  typedef Copy_access_tag    reorder_type;
  typedef Copy_access_tag    flex_type;
  typedef Copy_access_tag    copy_type;
  typedef Bogus_access_tag   bogus_type;
};



/// Choose access type for a given block and desired layout.
template <typename Block,
	  typename LP>
struct Choose_access
{
  typedef typename Block_layout<Block>::access_type access_type;
  typedef Access_demotion<access_type>              demotion_type;

  typedef
  choose_access::CA_General<demotion_type,
	     typename Block_layout<Block>::order_type,
	     typename Block_layout<Block>::pack_type,
	     typename Block_layout<Block>::complex_type,
             typename LP::order_type,
             typename LP::pack_type,
	     typename LP::complex_type> ca_type;

  typedef typename ca_type::type type;
  typedef typename ca_type::reason_type reason_type;
};

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_EXTDATA_HPP
