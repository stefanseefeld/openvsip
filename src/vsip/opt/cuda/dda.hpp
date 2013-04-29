/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_opt_cuda_dda_hpp_
#define vsip_opt_cuda_dda_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/device_storage.hpp>
#include <vsip/core/strided.hpp>
#include <vsip/dense.hpp>
#include <vsip/dda.hpp>
#include <vsip/opt/cuda/copy.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{
namespace dda
{
using vsip::dda::sync_policy;
using vsip::dda::in;
using vsip::dda::out;
using vsip::dda::inout;
using vsip::dda::impl::is_direct_ok;
using vsip::dda::impl::const_cast_;

/// Construct a unit-stride layout, avoiding copies, if possible:
///   - If Block provides DDA and has dense or aligned layout, use it.
///   - Else use dense.
template <typename Block>
struct Unit_stride_layout
{
private:
  typedef get_block_layout<typename remove_const<Block>::type> raw_type;

public:
  static dimension_type const dim = raw_type::dim;

  typedef typename raw_type::order_type    order_type;
  static pack_type const packing = 
    supports_dda<Block>::value &&
    is_packing_unit_stride<raw_type::packing>::value ? raw_type::packing : dense;
  static storage_format_type const storage_format = raw_type::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> layout_type;
};

/// Direct Data Access to Device storage from block.
/// By default, i.e. for non-specialized blocks, the Data
/// object needs to copy the block's data into device storage.
/// For blocks that support device storage internally, this storage
/// may be accessed directly, if its layout matches the desired layout.
/// For blocks with device storage, but non-matching layouts,
/// a copy is made in device memory.
template <typename Block,
	  sync_policy S,
	  typename L = typename Unit_stride_layout<Block>::layout_type,
	  bool HasDevice = has_device_storage<Block>::value>
class Data;

/// Specialization for blocks without internal device storage.
template <typename B, sync_policy S, typename L>
class Data<B, S, L, false> : public Non_copyable
{
  typedef L layout_type;
  typedef vsip::dda::Data<B, S, layout_type> data_type;
  typedef Device_storage<typename B::value_type, layout_type> storage_type;

public:
  static dimension_type const dim = layout_type::dim;
  typedef typename storage_type::type ptr_type;

  static int const ct_cost = 2;

  Data(B &block)
    : block_(block),
      data_(block_),
      storage_(Applied_layout<layout_type>(extent<dim>(block_)))
  { if (S & in) storage_.from_host(data_.ptr());}
  ~Data() { if (S & out) storage_.to_host(data_.ptr());}

  ptr_type ptr() { return storage_.ptr();}
  stride_type stride(dimension_type d) const { return data_.stride(d);}
  length_type size(dimension_type d) const { return data_.size(d);}
  length_type size() const { return storage_.total_size();}
    
private:
  B &block_;
  data_type data_;
  storage_type storage_;
};

/// Specialization for 'in' blocks without internal device storage.
template <typename B, typename L>
class Data<B, in, L, false> : public Non_copyable
{
  typedef L layout_type;
  typedef vsip::dda::Data<B, in, layout_type> data_type;
  typedef Device_storage<typename B::value_type, layout_type> storage_type;

public:
  static dimension_type const dim = layout_type::dim;
  typedef typename storage_type::const_type ptr_type;

  static int const ct_cost = 2;

  Data(B const &block)
    : block_(block),
      data_(block_),
      storage_(Applied_layout<layout_type>(extent<dim>(block_)))
  { storage_.from_host(data_.ptr());}
  ~Data() {}

  ptr_type ptr() { return storage_.ptr();}
  stride_type stride(dimension_type d) const { return data_.stride(d);}
  length_type size(dimension_type d) const { return data_.size(d);}
  length_type size() const { return storage_.total_size();}
    
private:
  B const &block_;
  data_type data_;
  storage_type storage_;
};

/// Specialization for blocks with internal device storage.
/// Device memory is handled by the block internally.
template <typename B, sync_policy S, typename L>
class Data<B, S, L, true> : public Non_copyable
{
  typedef typename B::value_type value_type;
public:
  static dimension_type const dim = L::dim;
  typedef typename Storage<L::storage_format, value_type>::type ptr_type;

  static int const ct_cost = 0;

  Data(B &block) : block_(block) {}

  ptr_type ptr()
  {
    // For an 'out' sync policy, invalidate host data to avoid data movement.
    if (!(S & in)) block_.invalidate_host();
    return block_.device_ptr();
  }
  stride_type stride(dimension_type d) const 
  { return block_.device_stride(dim, d);}
  length_type size(dimension_type d) const { return block_.size(dim, d);}
  length_type size() const { return block_.size();}
    
private:
  B &block_;
};

/// Specialization for 'in' blocks with internal device storage.
/// Device memory is handled by the block internally.
template <typename B, typename L>
class Data<B, in, L, true> : public Non_copyable
{
  typedef typename B::value_type value_type;
public:
  static dimension_type const dim = L::dim;
  typedef typename Storage<L::storage_format, value_type>::const_type ptr_type;

  static int const ct_cost = 0;

  Data(B const &block) : block_(block) {}
  ptr_type ptr() { return block_.device_ptr();}
  stride_type stride(dimension_type d) const 
  { return block_.device_stride(dim, d);}
  length_type size(dimension_type d) const { return block_.size(dim, d);}
  length_type size() const { return block_.size();}
    
private:
  B const &block_;
};

/// Copy needs to rearrange data in two ways:
/// It needs to remap between differing dimension orders,
/// and it needs to map between split and interleaved complex.
///
/// Right now only float and complex<float> variants are implemented.
/// The primary template is required since Data is instantiated
/// for types other than float and complex<float>, but it should never
/// be used.
template <typename Block,
	  typename T = typename Block::value_type,
	  dimension_type D = Block::dim,
	  typename O = typename Block::order_type>
struct Copy
{
  typedef Rt_layout<D> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<T, layout_type> storage_type;
  static void exec(Block const &, storage_type &) { assert(0);}
  static void exec(storage_type const &, Block &) { assert(0);}
};

template <typename B>
struct Copy<B, complex<float>, 1, tuple<0,1,2> >
{
  typedef Rt_layout<1> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<complex<float>, layout_type> storage_type;
  static void exec(B const &b, storage_type &s) 
  {
    if (s.storage_format() == split_complex)
      copy(b.device_ptr(), s.ptr().as_split(), s.size(0));
    else
      copy(b.device_ptr(), s.ptr().as_inter(), s.size(0));
  }
  static void exec(storage_type const &s, B &b)
  {
    if (s.storage_format() == split_complex)
      copy(s.ptr().as_split(), b.device_ptr(), s.size(0));
    else
      copy(s.ptr().as_inter(), b.device_ptr(), s.size(0));
  }
};

template <typename B>
struct Copy<B, float, 2, tuple<0,1,2> >
{
  typedef Rt_layout<2> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<float, layout_type> storage_type;
  static void exec(B const &b, storage_type &s)
  {
    if (s.stride(0) == 1) // to column-major
      transpose(b.device_ptr(), s.ptr().as_real(),
		b.size(2, 1), b.size(2, 0));
    else
      copy(b.device_ptr(), b.device_stride(2, 0),
	   s.ptr().as_real(), s.stride(0),
	   b.size(2, 0), b.size(2, 1));
  }
  static void exec(storage_type const &s, B &b)
  {
    if (s.stride(0) == 1) // from column-major
      transpose(s.ptr().as_real(), b.device_ptr(),
		b.size(2, 0), b.size(2, 1));
    else
      copy(s.ptr().as_real(), s.stride(0),
	   b.device_ptr(), b.device_stride(2, 0),
	   b.size(2, 0), b.size(2, 1));
  }
};

template <typename B>
struct Copy<B, complex<float>, 2, tuple<0,1,2> >
{
  typedef Rt_layout<2> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<complex<float>, layout_type> storage_type;
  static void exec(B const &b, storage_type &s)
  {
    if (s.stride(0) == 1) // to column-major
    {
      if (s.storage_format() == interleaved_complex)
       	transpose(b.device_ptr(), s.ptr().as_inter(),
       		  b.size(2, 1), b.size(2, 0));
      else
       	transpose(b.device_ptr(), s.ptr().as_split(),
      		  b.size(2, 1), b.size(2, 0));
    }
    else // to row-major
    {
      if (s.storage_format() == interleaved_complex)
	copy(b.device_ptr(), b.device_stride(2, 0),
	     s.ptr().as_inter(), s.stride(0),
	     b.size(2, 0), b.size(2, 1));
      else
	copy(b.device_ptr(), b.device_stride(2, 0),
	     s.ptr().as_split(), s.stride(0),
	     b.size(2, 0), b.size(2, 1));
    }
  }
  static void exec(storage_type const &s, B &b)
  {
    if (s.stride(0) == 1) // from column-major
    {
      if (s.storage_format() == interleaved_complex)
       	transpose(s.ptr().as_inter(), b.device_ptr(),
       		  b.size(2, 0), b.size(2, 1));
      else
       	transpose(s.ptr().as_split(), b.device_ptr(),
       		  b.size(2, 0), b.size(2, 1));
    }
    else // from row-major
    {
      if (s.storage_format() == interleaved_complex)
	copy(s.ptr().as_inter(), s.stride(0),
	     b.device_ptr(), b.device_stride(2, 0),
	     b.size(2, 0), b.size(2, 1));
      else
	copy(s.ptr().as_split(), s.stride(0),
	     b.device_ptr(), b.device_stride(2, 0),
	     b.size(2, 0), b.size(2, 1));
    }
  }
};

template <typename B>
struct Copy<B, float, 2, tuple<1,0,2> >
{
  typedef Rt_layout<2> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<float, layout_type> storage_type;
  static void exec(B const &b, storage_type &s)
  {
    if (s.stride(1) == 1) // to row-major
      transpose(b.device_ptr(), s.ptr().as_real(),
		b.size(2, 0), b.size(2, 1));
    else // to col-major
      copy(b.device_ptr(), b.device_stride(2, 1),
	   s.ptr().as_real(), s.stride(1),
	   b.size(2, 1), b.size(2, 0));
  }
  static void exec(storage_type const &s, B &b)
  {
    if (s.stride(1) == 1) // from row-major
      transpose(s.ptr().as_real(), b.device_ptr(),
		b.size(2, 1), b.size(2, 0));
    else // from row-major
      copy(s.ptr().as_real(), s.stride(1),
	   b.device_ptr(), b.device_stride(2, 1),
	   b.size(2, 1), b.size(2, 0));
  }
};

template <typename B>
struct Copy<B, complex<float>, 2, tuple<1,0,2> >
{
  typedef Rt_layout<2> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<complex<float>, layout_type> storage_type;
  static void exec(B const &b, storage_type &s)
  {
    if (s.stride(1) == 1) // to row-major
    {
      if (s.storage_format() == interleaved_complex)
       	transpose(b.device_ptr(), s.ptr().as_inter(),
       		  b.size(2, 0), b.size(2, 1));
      else
       	transpose(b.device_ptr(), s.ptr().as_split(),
      		  b.size(2, 0), b.size(2, 1));
    }
    else // to col-major
    {
      if (s.storage_format() == interleaved_complex)
	copy(b.device_ptr(), b.device_stride(2, 1),
	     s.ptr().as_inter(), s.stride(1),
	     b.size(2, 1), b.size(2, 0));
      else
	copy(b.device_ptr(), b.device_stride(2, 1),
	     s.ptr().as_split(), s.stride(1),
	     b.size(2, 1), b.size(2, 0));
    }
  }
  static void exec(storage_type const &s, B &b)
  {
    if (s.stride(1) == 1) // from row-major
    {
      if (s.storage_format() == interleaved_complex)
       	transpose(s.ptr().as_inter(), b.device_ptr(),
       		  b.size(2, 1), b.size(2, 0));
      else
       	transpose(s.ptr().as_split(), b.device_ptr(),
       		  b.size(2, 1), b.size(2, 0));
    }
    else // from col-major
    {
      if (s.storage_format() == interleaved_complex)
	copy(s.ptr().as_inter(), s.stride(1),
	     b.device_ptr(), b.device_stride(2, 1),
	     b.size(2, 1), b.size(2, 0));
      else
	copy(s.ptr().as_split(), s.stride(1),
	     b.device_ptr(), b.device_stride(2, 1),
	     b.size(2, 1), b.size(2, 0));
    }
  }
};

/// Specialization for runtime layout policy.
/// A layout mismatch results in copying done in host memory
template <typename B, sync_policy S, dimension_type D>
class Data<B, S, Rt_layout<D>, false> : public Non_copyable
{
  typedef Rt_layout<B::dim> layout_type;
  typedef Device_storage<typename B::value_type, layout_type> storage_type;

public:
  static dimension_type const dim = layout_type::dim;
  typedef typename storage_type::type ptr_type;

  static int const ct_cost = 2;

  Data(B &block, layout_type const &rtl, ptr_type buffer = ptr_type())
    : block_(block),
      data_(block_, rtl),
      storage_(Applied_layout<layout_type>(rtl, extent<dim>(block_)), false, buffer)
  { if (S & in) storage_.from_host(data_.ptr());}
  ~Data() { if (S & out) storage_.to_host(data_.ptr());}

  ptr_type ptr() { return storage_.ptr();}
  stride_type stride(dimension_type d) const { return storage_.stride(d);}
  length_type size(dimension_type d) const { return storage_.size(d);}
  length_type size() const { return storage_.total_size();}
    
private:
  B &block_;
  vsip::impl::Rt_data<B, S> data_;
  storage_type storage_;
};

/// Specialization for runtime layout policy.
/// A layout mismatch results in copying done in host memory
template <typename B, dimension_type D>
class Data<B, in, Rt_layout<D>, false> : public Non_copyable
{
  typedef Rt_layout<B::dim> layout_type;
  typedef Device_storage<typename B::value_type, layout_type> storage_type;

public:
  static dimension_type const dim = layout_type::dim;
  typedef typename storage_type::const_type ptr_type;
  typedef typename storage_type::type non_const_ptr_type;

  static int const ct_cost = 2;

  Data(B const &block, layout_type const &rtl, non_const_ptr_type buffer = non_const_ptr_type())
    : block_(block),
      data_(block_, rtl),
      storage_(Applied_layout<layout_type>(rtl, extent<dim>(block_)), false, buffer)
  { storage_.from_host(data_.ptr());}
  ~Data() {}

  ptr_type ptr() { return storage_.ptr();}
  non_const_ptr_type non_const_ptr() { return const_cast_<non_const_ptr_type>(ptr());}
  stride_type stride(dimension_type d) const { return storage_.stride(d);}
  length_type size(dimension_type d) const { return storage_.size(d);}
  length_type size() const { return storage_.total_size();}
    
private:
  B const &block_;
  vsip::impl::Rt_data<B, in> data_;
  storage_type storage_;
};

/// Specialization for blocks with device storage, with runtime layout policy.
/// A layout mismatch results in copying done in device memory.
template <dimension_type D, sync_policy S, typename B>
class Data<B, S, Rt_layout<D>, true> : public Non_copyable
{
  typedef Rt_layout<D> layout_type;
  typedef Device_storage<typename B::value_type, layout_type> storage_type;
  typedef Applied_layout<layout_type> applied_layout_type;

public:
  static dimension_type const dim = D;
  typedef typename storage_type::type ptr_type;

  static int const ct_cost = 1;

  Data(B &block, layout_type const &rtl, ptr_type buffer = ptr_type())
    : block_(block),
      use_direct_(is_direct_ok(block_, rtl)),
      app_layout_(rtl, extent<D>(block_)),
      storage_(app_layout_, use_direct_, buffer)
  {
    // Sync block's data to external storage.
    if (!use_direct_ && S & in)
      Copy<B>::exec(block_, storage_);
  }
  ~Data()
  { 
    if (!use_direct_ && S & out) 
      Copy<B>::exec(storage_, block_);
  }

  ptr_type ptr() 
  {
    if (use_direct_)
    {
      // For an 'out' sync policy, invalidate host data to avoid data movement.
      if (!(S & in)) block_.invalidate_host();
      return block_.device_ptr();
    }
    else return storage_.ptr();
  }
  stride_type stride(dimension_type d) const 
  { return use_direct_ ? block_.device_stride(D, d) : app_layout_.stride(d);}
  length_type size(dimension_type d) const 
  { return use_direct_ ? block_.size(D, d) : app_layout_.size(d);}
  length_type size() const 
  { return use_direct_ ? block_.size() : app_layout_.total_size();}
    
private:
  B &block_;
  bool use_direct_;
  applied_layout_type const app_layout_;
  storage_type storage_;

};

/// Specialization for immutable blocks with device storage, with runtime layout policy.
/// A layout mismatch results in copying done in device memory.
template <dimension_type D, typename B>
class Data<B, in, Rt_layout<D>, true> : public Non_copyable
{
  typedef Rt_layout<D> layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef Device_storage<typename B::value_type, layout_type> storage_type;

public:
  static dimension_type const dim = D;
  typedef typename storage_type::const_type ptr_type;
  typedef typename storage_type::type non_const_ptr_type;

  static int const ct_cost = 1;

  Data(B const &block, layout_type const &rtl, non_const_ptr_type buffer = non_const_ptr_type())
    : block_(block),
      use_direct_(is_direct_ok(block_, rtl)),
      app_layout_(rtl, extent<D>(block_)),
      storage_(app_layout_, use_direct_, buffer)
  { if (!use_direct_) Copy<B>::exec(block_, storage_);}

  ptr_type ptr() 
  { return use_direct_ ? ptr_type(block_.device_ptr()) : storage_.ptr();}
  non_const_ptr_type non_const_ptr() { return const_cast_<non_const_ptr_type>(ptr());}
  stride_type stride(dimension_type d) const 
  { return use_direct_ ? block_.device_stride(D, d) : app_layout_.stride(d);}
  length_type size(dimension_type d) const 
  { return use_direct_ ? block_.size(D, d) : app_layout_.size(d);}
  length_type size() const 
  { return use_direct_ ? block_.size() : app_layout_.total_size();}
    
private:
  B const &block_;
  bool use_direct_;
  applied_layout_type const app_layout_;
  storage_type storage_;

};

/// Convenience class
template <typename B, sync_policy S>
class Rt_data : public Data<B, S, Rt_layout<B::dim> >
{
  typedef Rt_layout<B::dim> layout_type;
  typedef Data<B, S, layout_type> base_type;

public:
  typedef typename base_type::ptr_type ptr_type;

  Rt_data(B &block, layout_type const &rtl, ptr_type buffer = ptr_type()) : base_type(block, rtl, buffer) {}
};

/// Convenience class
template <typename B>
class Rt_data<B, in> : public Data<B, in, Rt_layout<B::dim> >
{
  typedef Rt_layout<B::dim> layout_type;
  typedef Data<B, in, layout_type> base_type;

public:
  typedef typename base_type::non_const_ptr_type non_const_ptr_type;

  Rt_data(B const &block, layout_type const &rtl, non_const_ptr_type buffer = non_const_ptr_type()) : base_type(block, rtl, buffer) {}
};

} // namespace vsip::impl::cuda::dda
} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

#endif
