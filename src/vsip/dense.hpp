//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_dense_hpp_
#define vsip_dense_hpp_

#include <stdexcept>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/layout.hpp>
#include <vsip/impl/dense_fwd.hpp>
#include <ovxx/strided.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/allocator.hpp>

namespace vsip
{
template <typename T, typename O>
class Dense<1, T, O>
  : public ovxx::Strided<1, T, Layout<1, O, dense, ovxx::default_storage_format<T>::value> >
{
  typedef ovxx::Strided<1, T, Layout<1, O, dense, ovxx::default_storage_format<T>::value> >
    base_type;
  typedef typename base_type::uT uT;
public:
  typedef typename base_type::map_type map_type;

  Dense(Domain<1> const& dom, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, map)
    {}

  Dense(Domain<1> const& dom, T value, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, value, map)
    {}

  Dense(Domain<1> const& dom,
	T*const          pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
    {}

  Dense(Domain<1> const& dom,
	uT*const         pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
    {}

  Dense(Domain<1> const& dom,
	uT*const         real_pointer,
	uT*const         imag_pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, real_pointer, imag_pointer, map)
  {}

  Dense(Domain<1> const& dom,
	std::pair<uT*,uT*> pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
    {}
};

/// Partial specialization of Dense class template for 1,2-dimension.
template <typename T, typename O>
class Dense<2, T, O>
  : public ovxx::Strided<2, T, Layout<2, O, dense, ovxx::default_storage_format<T>::value> >
{
  typedef ovxx::Strided<2, T, Layout<2, O, dense, ovxx::default_storage_format<T>::value> >
    base_type;
  typedef typename base_type::uT uT;
public:
  typedef typename base_type::map_type             map_type;
  typedef typename base_type::reference_type       reference_type;
  typedef typename base_type::const_reference_type const_reference_type;

  Dense(Domain<2> const& dom, map_type const& map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, map)
  {}

  Dense(Domain<2> const& dom, T value, map_type const& map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map)
  {}

  Dense(Domain<2> const &dom,
	T *const pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
  {}

  Dense(Domain<2> const &dom,
	uT *const pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
  {}

  Dense(Domain<2> const &dom,
	uT *const real_pointer,
	uT *const imag_pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, real_pointer, imag_pointer, map)
  {}

  Dense(Domain<2> const &dom,
	std::pair<uT*,uT*> pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
  {}
};

template <typename T, typename O>
class Dense<3, T, O>
  : public ovxx::Strided<3, T, Layout<3, O, dense, ovxx::default_storage_format<T>::value> >
{
  typedef ovxx::Strided<3, T, Layout<3, O, dense, ovxx::default_storage_format<T>::value> >
    base_type;
  typedef typename base_type::uT uT;
public:
  typedef typename base_type::map_type             map_type;
  typedef typename base_type::reference_type       reference_type;
  typedef typename base_type::const_reference_type const_reference_type;

  Dense(Domain<3> const& dom, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, map)
  {}

  Dense(Domain<3> const& dom, T value, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, value, map)
  {}

  Dense(Domain<3> const& dom, T *const pointer, map_type const &map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, pointer, map)
  {}

  Dense(Domain<3> const &dom,
	uT *const pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, pointer, map)
  {}

  Dense(Domain<3> const &dom,
	uT *const real_pointer,
	uT *const imag_pointer,
	map_type const &map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, real_pointer, imag_pointer, map)
  {}

  Dense(Domain<3> const &dom,
	std::pair<uT*,uT*> pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, pointer, map)
  {}
};

#ifdef OVXX_PARALLEL

template <typename T, typename O, typename M>
class Dense<1, T, O, M> : public ovxx::parallel::Distributed_block<Dense<1, T, O>, M>
{
  typedef ovxx::parallel::Distributed_block<Dense<1, T, O>, M> base_type;
  typedef typename base_type::uT uT;
public:
  typedef M map_type;

  Dense(Domain<1> const& dom, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, map)
    {}

  Dense(Domain<1> const& dom, T value, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, value, map)
    {}

  Dense(Domain<1> const& dom,
	T*const          pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
    {}

  Dense(Domain<1> const& dom,
	uT*const         pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
    {}

  Dense(Domain<1> const& dom,
	uT*const         real_pointer,
	uT*const         imag_pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, real_pointer, imag_pointer, map)
  {}

  Dense(Domain<1> const& dom,
	std::pair<uT*,uT*> pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
    {}
};

/// Partial specialization of Dense class template for 1,2-dimension.
template <typename T, typename O, typename M>
class Dense<2, T, O, M> : public ovxx::parallel::Distributed_block<Dense<2, T, O>, M>
{
  typedef ovxx::parallel::Distributed_block<Dense<2, T, O>, M> base_type;
  typedef typename base_type::uT uT;
public:
  typedef typename base_type::map_type             map_type;
  typedef typename base_type::reference_type       reference_type;
  typedef typename base_type::const_reference_type const_reference_type;

  // Constructors.
public:
  Dense(Domain<2> const& dom, map_type const& map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, map)
  {}

  Dense(Domain<2> const& dom, T value, map_type const& map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, value, map)
  {}

  Dense(Domain<2> const &dom,
	T *const pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
  {}

  Dense(Domain<2> const &dom,
	uT *const pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
  {}

  Dense(Domain<2> const &dom,
	uT *const real_pointer,
	uT *const imag_pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, real_pointer, imag_pointer, map)
  {}

  Dense(Domain<2> const &dom,
	std::pair<uT*,uT*> pointer,
	map_type const &map = map_type())
  VSIP_THROW((std::bad_alloc))
  : base_type(dom, pointer, map)
  {}
};

template <typename T, typename O, typename M>
class Dense<3, T, O, M> : public ovxx::parallel::Distributed_block<Dense<3, T, O>, M>
{
  typedef ovxx::parallel::Distributed_block<Dense<3, T, O>, M> base_type;
  typedef typename base_type::uT uT;
public:
  typedef typename base_type::map_type             map_type;
  typedef typename base_type::reference_type       reference_type;
  typedef typename base_type::const_reference_type const_reference_type;

  Dense(Domain<3> const& dom, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, map)
  {}

  Dense(Domain<3> const& dom, T value, map_type const& map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, value, map)
  {}

  Dense(Domain<3> const& dom, T *const pointer, map_type const &map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, pointer, map)
  {}

  Dense(Domain<3> const &dom,
	uT *const pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, pointer, map)
  {}

  Dense(Domain<3> const &dom,
	uT *const real_pointer,
	uT *const imag_pointer,
	map_type const &map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, real_pointer, imag_pointer, map)
  {}

  Dense(Domain<3> const &dom,
	std::pair<uT*,uT*> pointer,
	map_type const&  map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, pointer, map)
  {}
};

#endif

/// Specialize block layout trait for Dense blocks.
template <dimension_type D, typename T, typename O, typename M>
struct get_block_layout<Dense<D, T, O, M> >
  : get_block_layout<ovxx::Strided<D, T,
				   Layout<D, O, dense, ovxx::default_storage_format<T>::value>, M> >
{};

template <dimension_type D, typename T, typename O, typename M>
struct supports_dda<Dense<D, T, O, M> >
{ static bool const value = true;};

/// Overload of get_local_block for Dense with local map.
template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
Dense<Dim, T, OrderT, Local_map>&
get_local_block(Dense<Dim, T, OrderT, Local_map>& block)
{
  return block;
}

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
Dense<Dim, T, OrderT, Local_map> const&
get_local_block(Dense<Dim, T, OrderT, Local_map> const& block)
{
  return block;
}

/// Overload of get_local_block for Dense with distributed map.
template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
inline typename Dense<Dim, T, OrderT, MapT>::local_block_type&
get_local_block(Dense<Dim, T, OrderT, MapT> const& block)
{
  return block.get_local_block();
}

namespace impl {

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
inline typename Dense<Dim, T, OrderT, MapT>::proxy_local_block_type
get_local_proxy(
  Dense<Dim, T, OrderT, MapT> const& block,
  index_type                         sb)
{
  return block.impl_proxy_block(sb);
}

}

/// Assert that subblock is local to block (overload).
template <dimension_type D, typename T, typename O>
void assert_local(Dense<D, T, O, Local_map> const &, index_type sb)
{
  OVXX_PRECONDITION(sb == 0);
}

/// Assert that subblock is local to block (overload).
template <dimension_type D, typename T, typename O, typename M>
void assert_local(Dense<D, T, O, M> const &block, index_type sb)
{
  block.assert_local(sb);
}

} // namespace vsip

namespace ovxx 
{

template <dimension_type D, typename T, typename O, typename M>
struct is_simple_distributed_block<Dense<D, T, O, M> >
{
  static bool const value = true;
};

template <dimension_type D, typename T, typename O, typename M>
struct is_modifiable_block<Dense<D, T, O, M> >
{
  static bool const value = true;
};

template <dimension_type D, typename T, typename O>
struct distributed_local_block<vsip::Dense<D, T, O, Local_map> >
{
  typedef vsip::Dense<D, T, O, Local_map> type;
  typedef vsip::Dense<D, T, O, Local_map> proxy_type;
};

template <dimension_type D, typename T, typename O, typename M>
struct distributed_local_block<vsip::Dense<D, T, O, M> >
{
  typedef vsip::Dense<D, T, O, Local_map> type;
  typedef typename vsip::Dense<D, T, O, M>::proxy_local_block_type proxy_type;
};

namespace detail
{
/// Specialize lvalue accessor trait for Dense blocks.
/// Dense provides direct lvalue accessors via ref.
template <typename B,
	  bool use_proxy = is_split_block<B>::value>
struct dense_lvalue_factory_type;

template <typename B>
struct dense_lvalue_factory_type<B, false>
{
  typedef ref_factory<B> type;
  template <typename O>
  struct rebind { typedef ref_factory<O> type;};
};

template <typename B>
struct dense_lvalue_factory_type<B, true>
{
  typedef proxy_factory<B> type;
  template <typename O>
  struct rebind { typedef proxy_factory<O> type;};
};

} // namespace ovxx::detail

template <dimension_type D, typename T, typename O>
struct lvalue_factory_type<Dense<D, T, O, Local_map> >
  : detail::dense_lvalue_factory_type<Dense<D, T, O, Local_map> >
{};

}

#endif
