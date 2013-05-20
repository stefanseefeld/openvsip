//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_strided_hpp_
#define ovxx_strided_hpp_

#include <ovxx/support.hpp>
#include <ovxx/storage/block.hpp>
#include <vsip/impl/local_map.hpp>
#include <ovxx/parallel/distributed_block.hpp>
#include <stdexcept>

namespace ovxx
{

// Strided is a generalization of vsip::Dense ([block.dense]),
// providing slightly more control over the physical layout
// of the data.
template <dimension_type D,
	  typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename L = Layout<D, tuple<0,1,2>, dense, array>,
	  typename M = Local_map>
class Strided;

/// Specialization for 1D non-distributed Strided.
template <typename T, typename L>
class Strided<1, T, L, Local_map> : public stored_block<T, L>
{
  typedef stored_block<T, L> base_type;
protected:
  typedef typename base_type::uT uT;
public:
  typedef Local_map map_type;

  Strided(Domain<1> const &dom,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, map) {}

  Strided(Domain<1> const &dom, T value,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map) {}

  Strided(Domain<1> const &dom, T *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<1> const &dom, uT *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<1> const &dom, uT *const r, uT *const i,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, std::make_pair(r, i), map) {}

  Strided(Domain<1> const &dom, std::pair<uT*,uT*> ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}
};

/// Specialization for 2D non-distributed Strided.
template <typename T, typename L>
class Strided<2, T, L, Local_map> : public stored_block<T, L>
{
  typedef stored_block<T, L> base_type;
protected:
  typedef typename base_type::uT uT;
public:
  typedef Local_map map_type;

  Strided(Domain<2> const &dom,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, map) {}

  Strided(Domain<2> const &dom, T value,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map) {}

  Strided(Domain<2> const &dom, T *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<2> const &dom, uT *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<2> const &dom, uT *const r, uT *const i,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, std::make_pair(r, i), map) {}

  Strided(Domain<2> const &dom, std::pair<uT*,uT*> ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1);}

  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, val);}
};

/// Specialization for 3D non-distributed Strided.
template <typename T, typename L>
class Strided<3, T, L, Local_map> : public stored_block<T, L>
{
  typedef stored_block<T, L> base_type;
protected:
  typedef typename base_type::uT uT;
public:
  typedef Local_map map_type;

  Strided(Domain<3> const &dom,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, map) {}

  Strided(Domain<3> const &dom, T value,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map) {}

  Strided(Domain<3> const &dom, T *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<3> const &dom, uT *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<3> const &dom, uT *const r, uT *const i,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, std::make_pair(r, i), map) {}

  Strided(Domain<3> const &dom, std::pair<uT*,uT*> ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1, idx2);}

  void put(index_type idx0, index_type idx1, index_type idx2, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, idx2, val);}
};

/// Specialization for 1D distributed Strided.
template <typename T, typename L, typename M>
class Strided<1, T, L, M> : public parallel::distributed_block<Strided<1, T, L>, M>
{
  typedef parallel::distributed_block<Strided<1, T, L>, M> base_type;
protected:
  typedef typename base_type::uT uT;
public:
  typedef M map_type;

  Strided(Domain<1> const &dom,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, map) {}

  Strided(Domain<1> const &dom, T value,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map) {}

  Strided(Domain<1> const &dom, T *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<1> const &dom, uT *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<1> const &dom, uT *const r, uT *const i,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, std::make_pair(r, i), map) {}

  Strided(Domain<1> const &dom, std::pair<uT*,uT*> ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}
};

/// Specialization for 2D distributed Strided.
template <typename T, typename L, typename M>
class Strided<2, T, L, M> : public parallel::distributed_block<Strided<2, T, L>, M>
{
  typedef parallel::distributed_block<Strided<2, T, L>, M> base_type;
protected:
  typedef typename base_type::uT uT;
public:
  typedef M map_type;

  Strided(Domain<2> const &dom,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, map) {}

  Strided(Domain<2> const &dom, T value,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map) {}

  Strided(Domain<2> const &dom, T *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<2> const &dom, uT *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<2> const &dom, uT *const r, uT *const i,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, std::make_pair(r, i), map) {}

  Strided(Domain<2> const &dom, std::pair<uT*,uT*> ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1);}

  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, val);}
};

/// Specialization for 3D distributed Strided.
template <typename T, typename L, typename M>
class Strided<3, T, L, M> : public parallel::distributed_block<Strided<3, T, L>, M>
{
  typedef parallel::distributed_block<Strided<3, T, L>, M> base_type;
protected:
  typedef typename base_type::uT uT;
public:
  typedef M map_type;

  Strided(Domain<3> const &dom,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, map) {}

  Strided(Domain<3> const &dom, T value,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map) {}

  Strided(Domain<3> const &dom, T *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<3> const &dom, uT *const ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  Strided(Domain<3> const &dom, uT *const r, uT *const i,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, std::make_pair(r, i), map) {}

  Strided(Domain<3> const &dom, std::pair<uT*,uT*> ptr,
	  map_type const &map = map_type()) VSIP_THROW((std::bad_alloc))
  : base_type(dom, ptr, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1, idx2);}

  void put(index_type idx0, index_type idx1, index_type idx2, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, idx2, val);}
};

template <dimension_type D, typename T, typename L, typename M>
struct is_modifiable_block<ovxx::Strided<D, T, L, M> >
{
  static bool const value = true;
};

template <dimension_type D, typename T, typename L>
struct distributed_local_block<Strided<D, T, L> >
{
  typedef Strided<D, T, L> type;
  typedef Strided<D, T, L> proxy_type;
};

template <dimension_type D, typename T, typename L, typename M>
struct distributed_local_block<Strided<D, T, L, M> >
{
  typedef Strided<D, T, L, Local_map> type;
  typedef typename Strided<D, T, L, M>::proxy_local_block_type proxy_type;
};

template <dimension_type D, typename T, typename L, typename M>
inline typename Strided<D, T, L, M>::local_block_type &
get_local_block(Strided<D, T, L, M> const &block)
{
  return block.get_local_block();
}

namespace detail
{
/// Specialize lvalue accessor trait for Strided blocks.
/// Strided provides direct lvalue accessors via ref unless data
/// are stored as split-complex.
template <typename B,
	  dimension_type D,
	  bool use_proxy = is_split_block<B>::value>
struct strided_lvalue_factory_type;

template <typename B, dimension_type D>
struct strided_lvalue_factory_type<B, D, false>
{
  typedef ref_factory<B> type;
  template <typename O>
  struct rebind { typedef ref_factory<O> type;};
};

template <typename B, dimension_type D>
struct strided_lvalue_factory_type<B, D, true>
{
  typedef proxy_factory<B, D> type;
  template <typename O>
  struct rebind { typedef proxy_factory<O, D> type;};
};

} // namespace ovxx::detail

template <dimension_type D, typename T, typename L>
struct lvalue_factory_type<Strided<D, T, L>, D>
  : detail::strided_lvalue_factory_type<Strided<D, T, L>, D>
{};

} // namespace ovxx

namespace vsip
{

/// Specialize block layout trait for Strided.
template <dimension_type D, typename T, typename L, typename M>
struct get_block_layout<ovxx::Strided<D, T, L, M> >
{
  static dimension_type const dim = D;
  typedef typename L::order_type order_type;
  static pack_type const packing = L::packing;
  static storage_format_type const storage_format = L::storage_format;
  typedef L type;
};

template <dimension_type D, typename T, typename L, typename M>
struct supports_dda<ovxx::Strided<D, T, L, M> >
{ static bool const value = true;};

} // namespace vsip

#endif
