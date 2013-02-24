/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef vsip_core_strided_hpp_
#define vsip_core_strided_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#ifdef VSIP_IMPL_HAVE_CUDA
# include <vsip/opt/cuda/stored_block.hpp>
#else
# include <vsip/core/stored_block.hpp>
#endif
#include <vsip/core/parallel/distributed_block.hpp>
#include <stdexcept>

namespace vsip
{
namespace impl
{
#ifdef VSIP_IMPL_HAVE_CUDA
using cuda::Stored_block;
#endif

/// A Strided is a modifiable, allocatable x-dimensional block
/// or 1,x-dimensional block, for a fixed x, that explicitly stores
/// one value for each Index in its domain.
///
/// Template parameters:
///   :D: the block's dimension,
///   :T: the block's value-type,
///   :L: the block's layout policy, encapsulating dimension-ordering,
///        packing format, and complex format,
///   :M: the block's map type.
template <dimension_type D,
	  typename T = VSIP_DEFAULT_VALUE_TYPE,
	  typename L = Layout<D,
			      typename Row_major<D>::type,
			      Stride_unit_dense,
			      Cmplx_inter_fmt>,
	  typename M = Local_map>
class Strided;

/// Specialization for 1D non-distributed Strided.
template <typename T, typename L>
class Strided<1, T, L, Local_map> : public Stored_block<T, L>
{
  typedef Stored_block<T, L> base_type;
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
  : base_type(dom, r, i, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}
};

/// Specialization for 2D non-distributed Strided.
template <typename T, typename L>
class Strided<2, T, L, Local_map> : public Stored_block<T, L>
{
  typedef Stored_block<T, L> base_type;
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
  : base_type(dom, r, i, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1);}

  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, val);}
};

/// Specialization for 3D non-distributed Strided.
template <typename T, typename L>
class Strided<3, T, L, Local_map> : public Stored_block<T, L>
{
  typedef Stored_block<T, L> base_type;
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
  : base_type(dom, r, i, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1, idx2);}

  void put(index_type idx0, index_type idx1, index_type idx2, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, idx2, val);}
};

/// Specialization for 1D distributed Strided.
template <typename T, typename L, typename M>
class Strided<1, T, L, M> : public Distributed_block<Strided<1, T, L>, M>
{
  typedef Distributed_block<Strided<1, T, L>, M> base_type;
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
  : base_type(dom, r, i, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}
};

/// Specialization for 2D distributed Strided.
template <typename T, typename L, typename M>
class Strided<2, T, L, M> : public Distributed_block<Strided<2, T, L>, M>
{
  typedef Distributed_block<Strided<2, T, L>, M> base_type;
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
  : base_type(dom, r, i, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1);}

  void put(index_type idx0, index_type idx1, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, val);}
};

/// Specialization for 3D distributed Strided.
template <typename T, typename L, typename M>
class Strided<3, T, L, M> : public Distributed_block<Strided<3, T, L>, M>
{
  typedef Distributed_block<Strided<3, T, L>, M> base_type;
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
  : base_type(dom, r, i, map) {}

  T get(index_type idx) const VSIP_NOTHROW { return base_type::get(idx);}

  void put(index_type idx, T val) VSIP_NOTHROW { base_type::put(idx, val);}

  T get(index_type idx0, index_type idx1, index_type idx2) const VSIP_NOTHROW 
  { return base_type::get(idx0, idx1, idx2);}

  void put(index_type idx0, index_type idx1, index_type idx2, T val) VSIP_NOTHROW
  { base_type::put(idx0, idx1, idx2, val);}
};

/// Specialize block layout trait for Strided.
template <dimension_type D, typename T, typename L, typename M>
struct Block_layout<Strided<D, T, L, M> >
{
  static dimension_type const dim = D;
  typedef Direct_access_tag access_type;
  typedef L layout_type;
  typedef typename layout_type::order_type order_type;
  typedef typename layout_type::pack_type pack_type;
  typedef typename layout_type::complex_type complex_type;
};

template <dimension_type D, typename T, typename L, typename M>
struct Is_pas_block<Strided<D, T, L, M> >
  : Is_pas_block<Distributed_block<Strided<D, T, L>, M> >
{};

template <dimension_type D, typename T, typename L>
struct Is_pas_block<Strided<D, T, L> >
{
  static bool const value = false;
};

template <dimension_type D, typename T, typename L, typename M>
struct Is_modifiable_block<Strided<D, T, L, M> >
{
  static bool const value = true;
};

#ifdef VSIP_IMPL_HAVE_CUDA
namespace cuda
{
template <dimension_type D, typename T, typename L>
struct has_device_storage<Strided<D, T, L> > { static bool const value = true;};

}
#endif

} // namespace vsip::impl
} // namespace vsip

#endif
