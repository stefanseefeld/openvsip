//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_storage_block_hpp_
#define ovxx_storage_block_hpp_

#include <ovxx/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/refcounted.hpp>
#include <ovxx/layout.hpp>
#include <ovxx/storage.hpp>
#include <ovxx/block_traits.hpp>
#include <vsip/impl/local_map.hpp>
#include <stdexcept>

namespace ovxx
{

// Report the default storage-format for blocks
// of value-type 'T'. This can only be 'array'
// for non-complex value-types.
template <typename T>
struct default_storage_format
{
  static storage_format_type const value = array;
};

// Report the default storage-format for complex
// blocks, which is set at configure-time.
template <typename T>
struct default_storage_format<complex<T> >
{
#if OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT
  static storage_format_type const value = split_complex;
#else
  static storage_format_type const value = array;
#endif
};

// An allocatable / modifiable non-distributed block type
// that stores its elements internally. This block supports
// the user-storage API ([block.dense.userdata]).
// The implementation uses a storage-manager, which manages
// potentially multiple storage areas, supporting heterogeneous
// memory. Data are replicated and moved on-demand.
template <typename T, typename L>
class stored_block : public refcounted<stored_block<T, L> >
{
  typedef storage<T, L::storage_format> storage_type;
  typedef storage_manager<T, L::storage_format> smanager_type;
protected:
  enum private_type {};
  typedef typename detail::complex_value_type<T, private_type>::type uT;
public:
  typedef T value_type;
  typedef T &reference_type;
  typedef T const &const_reference_type;

  typedef L layout_type;
  typedef Applied_layout<layout_type> applied_layout_type;
  typedef typename L::order_type order_type;

  typedef typename smanager_type::ptr_type ptr_type;
  typedef typename smanager_type::const_ptr_type const_ptr_type;

  typedef Local_map map_type;

  static dimension_type const dim = L::dim;
  static storage_format_type const storage_format = L::storage_format;

  stored_block(Domain<dim> const &dom, map_type const &map = map_type())
    : layout_(dom),
      smanager_(map.impl_allocator(), layout_.total_size()),
      map_(map)
  {
    map_.impl_apply(dom);
  }

  stored_block(Domain<dim> const &dom, T value, map_type const &map = map_type())
    : layout_(dom),
      smanager_(map.impl_allocator(), layout_.total_size()),
      map_(map)
  {
    map_.impl_apply(dom);
    for (index_type i = 0; i < layout_.total_size(); ++i)
      smanager_.put(i, value);
  }

  stored_block(Domain<dim> const &dom,
	       T *const ptr,
	       map_type const &map = map_type())
    : layout_(dom),
      smanager_(map.impl_allocator(), layout_.total_size(),
		ptr),
      map_(map)
  {
  }

  stored_block(Domain<dim> const &dom,
	       uT *const ptr,
	       map_type const &map = map_type())
    : layout_(dom),
      smanager_(map.impl_allocator(), layout_.total_size(),
		ptr),
      map_(map)
  {
  }

  stored_block(Domain<dim> const &dom,
	       std::pair<uT*,uT*> ptr,
	       map_type const &map = map_type())
    : layout_(dom),
      smanager_(map.impl_allocator(), layout_.total_size(), ptr),
      map_(map)
  {
  }

  ~stored_block() {}

  void rebind(T *ptr)
  {
    smanager_.rebind(ptr, layout_.total_size());
  }

  void rebind(uT *ptr)
  {
    smanager_.rebind(ptr, layout_.total_size());
  }

  void rebind(uT *rp, uT *ip)
  {
    smanager_.rebind(std::make_pair(rp, ip), layout_.total_size());
  }
  void rebind(std::pair<uT*,uT*> ptr) { smanager_.rebind(ptr, layout_.total_size());}

  void rebind(T *ptr, Domain<dim> const &dom)
  {
    layout_ = applied_layout_type(dom);
    smanager_.rebind(ptr, layout_.total_size());
  }

  void rebind(uT *ptr, Domain<dim> const &dom)
  {
    layout_ = applied_layout_type(dom);
    smanager_.rebind(ptr, layout_.total_size());
  }
  void rebind(uT *rp, uT *ip, Domain<dim> const &dom)
  {
    layout_ = applied_layout_type(dom);
    smanager_.rebind(std::make_pair(rp, ip), layout_.total_size());
  }
  void rebind(std::pair<uT*,uT*> ptr, Domain<dim> const &dom)
  {
    layout_ = applied_layout_type(dom);
    smanager_.rebind(ptr, layout_.total_size());
  }

  void find(T *&ptr) VSIP_NOTHROW
  {
    smanager_.find(ptr);
  }
  void find(uT *&ptr) VSIP_NOTHROW
  {
    smanager_.find(ptr);
  }
  void find(uT *&rp, uT *&ip) VSIP_NOTHROW
  {
    std::pair<uT*,uT*> ptr;
    smanager_.find(ptr);
    rp = ptr.first;
    ip = ptr.second;
  }
  void find(std::pair<uT*,uT*> &ptr) VSIP_NOTHROW { smanager_.find(ptr);}

  void admit(bool update = true) VSIP_NOTHROW
  {
    smanager_.admit(update);
  }

  void release(bool update = true) VSIP_NOTHROW
  {
    smanager_.release(update);
  }

  void release(bool update, T *&ptr) VSIP_NOTHROW
  {
    smanager_.release(update, ptr);
  }

  void release(bool update, uT *&ptr) VSIP_NOTHROW
  {
    smanager_.release(update, ptr);
  }

  void release(bool update, uT *&rp, uT *&ip) VSIP_NOTHROW
  {
    std::pair<uT*,uT*> ptr;
    smanager_.release(update, ptr);
    rp = ptr.first;
    ip = ptr.second;
  }

  void release(bool update, std::pair<uT*,uT*> &ptr) VSIP_NOTHROW
  { smanager_.release(update, ptr);}

  user_storage_type user_storage() const VSIP_NOTHROW
  { return smanager_.user_storage();}
  bool admitted() const VSIP_NOTHROW { return smanager_.admitted();}

  map_type const &map() const VSIP_NOTHROW { return map_;}

  length_type size() const VSIP_NOTHROW
  {
    length_type retval = layout_.size(0);
    for (dimension_type d = 1; d < dim; ++d)
      retval *= layout_.size(d);
    return retval;
  }

  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION((block_dim == 1 || block_dim == dim) && (d < block_dim));

    if (block_dim == 1) return size();
    else return layout_.size(d);
  }

  value_type get(index_type idx) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx < size());
    return smanager_.get(idx);
  }
  value_type get(index_type idx0, index_type idx1) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx0 < layout_.size(0) && idx1 < layout_.size(1));
    return smanager_.get(layout_.index(idx0, idx1));
  }
  value_type get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(idx0 < layout_.size(0) && idx1 < layout_.size(1) &&
		      idx2 < layout_.size(2));
    return smanager_.get(layout_.index(idx0, idx1, idx2));
  }

  void put(index_type i, T v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    smanager_.put(i, v);
  }
  void put(index_type i, index_type j, T v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1));
    smanager_.put(layout_.index(i, j), v);
  }
  void put(index_type i, index_type j, index_type k, T v)
    VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1) &&
		      k < layout_.size(2));
    smanager_.put(layout_.index(i, j, k), v);
  }

  reference_type ref(index_type i) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    return smanager_.at(i);
  }
  const_reference_type ref(index_type i) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    return smanager_.at(i);
  }
  reference_type ref(index_type i, index_type j)
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1));
    return smanager_.at(layout_.index(i, j));
  }
  const_reference_type ref(index_type i, index_type j) const
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1));
    return smanager_.at(layout_.index(i, j));
  }
  reference_type ref(index_type i, index_type j, index_type k)
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1) && k < layout_.size(2));
    return smanager_.at(layout_.index(i, j, k));
  }
  const_reference_type ref(index_type i, index_type j, index_type k) const
  {
    OVXX_PRECONDITION(i < layout_.size(0) && j < layout_.size(1) && k < layout_.size(2));
    return smanager_.at(layout_.index(i, j, k));
  }

  ptr_type ptr() { return smanager_.ptr();}
  const_ptr_type ptr() const { return smanager_.ptr();}
#if OVXX_HAVE_OPENCL
  opencl::buffer buffer() { return smanager_.buffer();}
  opencl::buffer buffer() const { return smanager_.buffer();}
#endif
  stride_type stride(dimension_type block_dim, dimension_type d) const
  {
    OVXX_PRECONDITION(block_dim == 1 || block_dim == dim);
    OVXX_PRECONDITION(d < dim);
    if (block_dim == 1) return 1;
    else return layout_.stride(d);
  }

private:
  applied_layout_type layout_;
  smanager_type smanager_;
  map_type map_;
};

} // namespace ovxx

#endif
