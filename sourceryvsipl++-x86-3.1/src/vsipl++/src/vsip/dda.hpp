/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */
#ifndef vsip_dda_hpp_
#define vsip_dda_hpp_

#include <vsip/core/c++0x.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/dda.hpp>
#if !VSIP_IMPL_REF_IMPL
# include <vsip/opt/dda.hpp>
# include <vsip/opt/rt_dda.hpp>
#endif

namespace vsip
{
namespace dda
{

/// Determine desired block layout.
///
/// For a block with direct access, the desired layout is the same
/// as the block's layout (get_block_layout).
///
/// For a block with copy access, the desired layout adjusts the
/// pack type to be dense, so that the block can be copied into
/// contiguous memory.
template <typename Block>
struct dda_block_layout
{
private:
  typedef typename vsip::impl::remove_const<Block>::type block_type;
  typedef typename get_block_layout<block_type>::type block_layout_type;

public:
  static dimension_type const dim = block_layout_type::dim;

  typedef typename block_layout_type::order_type    order_type;
  static pack_type const packing = 
    supports_dda<block_type>::value
    ? block_layout_type::packing
    : (vsip::impl::is_packing_aligned<block_layout_type::packing>::value
       ? block_layout_type::packing
       : dense);
  static storage_format_type const storage_format = block_layout_type::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> layout_type;
};

enum sync_policy
{
  /// Synchronize input block on DDA creation
  in = 0x01,
  /// Synchronize output block on DDA destruction
  out = 0x02,
  /// Synchronize inout block in both directions
  inout = in | out // 0x03
};

/// Direct (block) data accessor.
///
/// Template parameters
///
///   :Block:  the block type.
///   :Sync:   a sync-policy (`in`, `out`, or `inout`)
///   :Layout: the desired layout policy for the data access.
///
/// The general case covers modifiable (`inout` & `out`) blocks
template <typename Block,
	  sync_policy Sync,
	  typename Layout = typename dda_block_layout<Block>::layout_type>
class Data : vsip::impl::Non_copyable,
             vsip::impl::Assert_modifiable_block<Block>
{
  typedef typename Block::value_type value_type;
  typedef typename impl::Choose_access<Block, Layout>::type access_type;

public:
  // FIXME: the next two types are non-standard
  typedef vsip::impl::Storage<Layout::storage_format, value_type> storage_type;
  typedef typename storage_type::alloc_type element_type;

  typedef impl::Accessor<Block, Layout, access_type> backend_type;

  typedef typename backend_type::const_ptr_type const_ptr_type;
  typedef typename backend_type::ptr_type ptr_type;

  static int   const ct_cost = backend_type::CT_Cost;

  Data(Block &block, ptr_type buffer = ptr_type())
    : block_(block), backend_(block, buffer)
  { backend_.begin(&block_, (Sync & in) != 0);}

  ~Data() { backend_.end(&block_, (Sync & out) != 0);}

  ptr_type ptr() { return backend_.ptr(&block_);}
  const_ptr_type ptr() const { return backend_.ptr(&block_);}
  stride_type stride(dimension_type d) const { return backend_.stride(&block_, d);}
  length_type size(dimension_type d) const { return backend_.size(&block_, d);}
  length_type size() const { return backend_.size(&block_);}
  int cost() const { return backend_.cost();}

private:
  Block &block_;
  backend_type backend_;
};

/// Specialization for `in` synchronization
template <typename Block, typename L>
class Data<Block, in, L> : vsip::impl::Non_copyable
{
  typedef typename vsip::impl::remove_const<Block>::type non_const_block_type;
  typedef typename Block::value_type value_type;
  typedef typename impl::Choose_access<Block, L>::type access_type;

public:
  // FIXME: the next two types are non-standard
  typedef vsip::impl::Storage<L::storage_format, value_type> storage_type;
  typedef typename storage_type::alloc_type element_type;

  typedef impl::Accessor<Block const, L, access_type> backend_type;

  typedef typename backend_type::non_const_ptr_type non_const_ptr_type;
  typedef typename backend_type::const_ptr_type ptr_type;

  static int   const ct_cost = backend_type::CT_Cost;

  Data(Block const &block, non_const_ptr_type buffer = non_const_ptr_type())
    : block_(block), backend_(const_cast<non_const_block_type &>(block), buffer)
  { backend_.begin(&block_, true);}

  ~Data() { backend_.end(&block_, false);}

  ptr_type ptr() const { return backend_.ptr(&block_);}
  non_const_ptr_type non_const_ptr() { return impl::const_cast_<non_const_ptr_type>(ptr());}
  stride_type stride(dimension_type d) const { return backend_.stride(&block_, d);}
  length_type size(dimension_type d) const { return backend_.size(&block_, d);}
  length_type size() const { return backend_.size(&block_);}
  int cost() const { return backend_.cost();}

private:
  Block const &block_;
  backend_type backend_;
};

/// Return the cost of accessing a block with a given layout.
template <typename L, typename Block>
inline int cost(Block const &block, L const &layout = L())
{
  typedef typename impl::Choose_access<Block, L>::type access_type;
  return impl::Accessor<Block, L, access_type>::cost(block, layout);
}

/// Return the cost of accessing a block
template <typename Block>
inline int cost(Block const &block)
{
  typedef typename get_block_layout<Block>::type layout_type;
  return cost<layout_type>(block);
}

#if !VSIP_IMPL_REF_IMPL

/// Specialization for modifiable blocks, using a runtime layout
template <typename B, sync_policy S, dimension_type D>
class Data<B, S, vsip::impl::Rt_layout<D> > : vsip::impl::Non_copyable
{
  typedef B block_type;

  typedef typename 
  vsip::impl::conditional<supports_dda<B>::value,
			  impl::Direct_access_tag,
			  impl::Copy_access_tag>::type access_type;
  typedef impl::Rt_Accessor<B, D, access_type> backend_type;
  typedef vsip::impl::Rt_layout<B::dim> layout_type;

  static bool  const CT_Mem_not_req   = backend_type::CT_Mem_not_req;
  static bool  const CT_Xfer_not_req  = backend_type::CT_Xfer_not_req;

public:

  typedef typename B::value_type value_type;
  typedef typename backend_type::non_const_ptr_type ptr_type;
  typedef typename backend_type::const_ptr_type const_ptr_type;

  static int const ct_cost = backend_type::CT_Cost;

  Data(block_type &block,
       layout_type const &rtl,
       ptr_type buffer = ptr_type(),
       length_type buffer_size = 0)
    : block_(block),
      backend_(block_, rtl, false, buffer, buffer_size)
    { backend_.begin(&block_, S & dda::in);}

  ~Data() { backend_.end(&block_, S & dda::out);}

  ptr_type ptr() { return backend_.ptr(&block_);}
  const_ptr_type ptr() const { return backend_.ptr(&block_);}
  stride_type  stride(dimension_type d) const { return backend_.stride(&block_, d);}
  length_type  size(dimension_type d) const { return backend_.size(&block_, d);}
  length_type  size() const { return backend_.size(&block_);}
  int cost() const { return backend_.cost();}

private:
  block_type &block_;
  backend_type backend_;
};

/// Specialization for unmodifiable data with runtime layout
template <typename B, dimension_type D>
class Data<B, in, vsip::impl::Rt_layout<D> > : vsip::impl::Non_copyable
{
  typedef B block_type;

  typedef typename 
  vsip::impl::conditional<supports_dda<B>::value,
			  impl::Direct_access_tag,
			  impl::Copy_access_tag>::type access_type;
  typedef impl::Rt_Accessor<B const, D, access_type> backend_type;
  typedef vsip::impl::Rt_layout<B::dim> layout_type;

  static bool  const CT_Mem_not_req   = backend_type::CT_Mem_not_req;
  static bool  const CT_Xfer_not_req  = backend_type::CT_Xfer_not_req;

public:

  typedef typename B::value_type value_type;
  typedef typename backend_type::non_const_ptr_type non_const_ptr_type;
  typedef typename backend_type::const_ptr_type ptr_type;

  static int const ct_cost = backend_type::CT_Cost;

  Data(block_type const &block,
       layout_type const &rtl,
       non_const_ptr_type buffer = non_const_ptr_type(),
       length_type buffer_size = 0)
    : block_(block),
      backend_(block_, rtl, false, buffer, buffer_size)
    { backend_.begin(&block_, true);}

  Data(block_type const &block,
       bool force_copy,
       layout_type const &rtl,
       non_const_ptr_type buffer = non_const_ptr_type(),
       length_type buffer_size = 0)
    : block_(block),
      backend_(block_, rtl, force_copy, buffer, buffer_size)
    { backend_.begin(&block_, true);}

  ~Data() { backend_.end(&block_, false);}

  ptr_type ptr() const { return backend_.ptr(&block_);}
  non_const_ptr_type non_const_ptr() { return impl::const_cast_<non_const_ptr_type>(ptr());}
  stride_type  stride(dimension_type d) const { return backend_.stride(&block_, d);}
  length_type  size(dimension_type d) const { return backend_.size(&block_, d);}
  length_type  size() const { return backend_.size(&block_);}
  int cost() const { return backend_.cost();}

private:
  block_type const &block_;
  backend_type backend_;
};

#endif

namespace impl
{
/// Return the number of bytes of memory required to access a block
/// with a given layout.
template <typename LP, typename Block>
size_t
mem_required(Block const& block, LP const &layout = LP())
{
  typedef typename Choose_access<Block, LP>::type access_type;
  return Accessor<Block, LP, access_type>::mem_required(block, layout);
}

/// Return whether a transfer is required to access a block with
/// a given layout.
template <typename LP, typename Block>
bool
xfer_required(Block const& block, LP const& layout = LP())
{
  typedef typename Choose_access<Block, LP>::type access_type;
  return Accessor<Block, LP, access_type>::xfer_required(block, layout);
}

/// Determine if an dda::Data object refers to a dense (contiguous,
/// unit-stride) region of memory.
template <typename OrderT, typename ExtT>
bool
is_ext_dense(vsip::dimension_type dim, ExtT const &ext)
{
  using vsip::dimension_type;
  using vsip::stride_type;

  dimension_type const dim0 = OrderT::impl_dim0;
  dimension_type const dim1 = OrderT::impl_dim1;
  dimension_type const dim2 = OrderT::impl_dim2;

  assert(dim <= VSIP_MAX_DIMENSION);

  if (dim == 1)
  {
    return (ext.stride(dim0) == 1);
  }
  else if (dim == 2)
  {
    return (ext.stride(dim1) == 1) &&
           (ext.stride(dim0) == static_cast<stride_type>(ext.size(dim1)) ||
	    ext.size(dim0) == 1);
  }
  else /*  if (dim == 2) */
  {
    return (ext.stride(dim2) == 1) &&
           (ext.stride(dim1) == static_cast<stride_type>(ext.size(dim2)) ||
	    (ext.size(dim0) == 1 && ext.size(dim1) == 1)) &&
           (ext.stride(dim0) == static_cast<stride_type>(ext.size(dim1)  *
							 ext.size(dim2)) ||
	    ext.size(dim0) == 1);
  }
}

} // namespace vsip::dda::impl
} // namespace vsip::dda

namespace impl
{
#if !VSIP_IMPL_REF_IMPL

template <typename B, dda::sync_policy S>
class Rt_data : public dda::Data<B, S, Rt_layout<B::dim> >
{
  typedef dda::Data<B, S, Rt_layout<B::dim> > base_type;
  typedef Rt_layout<B::dim> layout_type;

public:
  typedef typename base_type::ptr_type ptr_type;

  Rt_data(B &block,
	  layout_type const &rtl,
	  ptr_type buffer = ptr_type(),
	  length_type size = 0)
    : base_type(block, rtl, buffer, size) {}
};

template <typename B>
class Rt_data<B, dda::in> : public dda::Data<B, dda::in, Rt_layout<B::dim> >
{
  typedef dda::Data<B, dda::in, Rt_layout<B::dim> > base_type;
  typedef Rt_layout<B::dim> layout_type;

public:
  typedef typename base_type::non_const_ptr_type non_const_ptr_type;

  Rt_data(B const &block,
	  layout_type const &rtl,
	  non_const_ptr_type buffer = non_const_ptr_type(),
	  length_type size = 0)
    : base_type(block, rtl, buffer, size) {}

  Rt_data(B const &block,
	  bool force_copy,
	  layout_type const &rtl,
	  non_const_ptr_type buffer = non_const_ptr_type(),
	  length_type size = 0)
    : base_type(block, force_copy, rtl, buffer, size) {}
};

#endif

template <typename Block,
	  typename LP  = typename dda::dda_block_layout<Block>::layout_type>
class Persistent_data : Assert_proper_block<Block>
{
  typedef typename dda::impl::Choose_access<Block, LP>::type access_type;
  typedef dda::impl::Accessor<Block, LP, access_type> backend_type;

  static bool  const CT_Mem_not_req   = backend_type::CT_Mem_not_req;
  static bool  const CT_Xfer_not_req  = backend_type::CT_Xfer_not_req;

public:

  typedef typename Block::value_type value_type;

  typedef Storage<LP::storage_format, typename Block::value_type> storage_type;

  typedef typename storage_type::alloc_type element_type;

  typedef typename storage_type::type non_const_ptr_type;
  typedef typename storage_type::const_type const_ptr_type;
  typedef typename 
  vsip::impl::conditional<vsip::impl::is_modifiable_block<Block>::value,
			  non_const_ptr_type,
			  const_ptr_type>::type ptr_type;

  static int const ct_cost = backend_type::CT_Cost;

  Persistent_data(Block &block,
		  dda::sync_policy sync = dda::inout,
		  non_const_ptr_type buffer = non_const_ptr_type())
    : block_(block),
      backend_(block, buffer),
      sync_(sync)
    {}

  ~Persistent_data() {}
  void begin() { backend_.begin(&block_, (sync_ & dda::in) != 0);}
  void end() { backend_.end(&block_, (sync_ & dda::out) != 0);}

  ptr_type ptr() { return backend_.ptr(&block_);}
  const_ptr_type ptr() const { return backend_.ptr(&block_);}
  non_const_ptr_type non_const_ptr() 
  { return dda::impl::const_cast_<non_const_ptr_type>(backend_.ptr(&block_));}
  stride_type stride(dimension_type d) const { return backend_.stride(&block_, d);}
  length_type size(dimension_type d) const { return backend_.size(&block_, d);}
  length_type size() const { return backend_.size(&block_);}
  int cost() const { return backend_.cost();}

private:
  Block &block_;
  backend_type backend_;
  dda::sync_policy sync_;
};

} // namespace vsip::impl
} // namespace vsip

#endif
