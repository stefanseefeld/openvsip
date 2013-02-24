/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved. */

#ifndef VSIP_DENSE_HPP
#define VSIP_DENSE_HPP

#include <stdexcept>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/dense_fwd.hpp>
#include <vsip/core/dense_storage.hpp>
#include <vsip/core/user_storage.hpp>
#include <vsip/core/strided.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/parallel/local_map.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/memory_pool.hpp>
#include <vsip/core/profile.hpp>

namespace vsip
{
namespace impl
{ 
/// Complex storage format for dense blocks.
#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
storage_format_type const dense_complex_format = split_complex;
#else
storage_format_type const dense_complex_format = interleaved_complex;
#endif

} // namespace vsip::impl

template <typename T, typename O>
class Dense<1, T, O>
  : public impl::Strided<1, T, Layout<1, O, dense, impl::dense_complex_format> >
{
  typedef impl::Strided<1, T, Layout<1, O, dense, impl::dense_complex_format> >
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

  // Internal user storage constructor.
  Dense(Domain<1> const&             dom,
	impl::User_storage<T> const& data,
	map_type const&              map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, data, map)
    {}
};

/// Partial specialization of Dense class template for 1,2-dimension.
template <typename T, typename O>
class Dense<2, T, O>
  : public impl::Strided<2, T, Layout<2, O, dense, impl::dense_complex_format> >
{
  typedef impl::Strided<2, T, Layout<2, O, dense, impl::dense_complex_format> >
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

  reference_type impl_ref(index_type idx0, index_type idx1) VSIP_NOTHROW
  { return base_type::impl_ref(Index<2>(idx0, idx1));}

  const_reference_type impl_ref(index_type idx0, index_type idx1) const VSIP_NOTHROW
  { return base_type::impl_ref(Index<2>(idx0, idx1));}
};

template <typename T, typename O>
class Dense<3, T, O>
  : public impl::Strided<3, T, Layout<3, O, dense, impl::dense_complex_format> >
{
  typedef impl::Strided<3, T, Layout<3, O, dense, impl::dense_complex_format> >
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

  reference_type impl_ref(index_type idx0, index_type idx1, index_type idx2)
    VSIP_NOTHROW
  { return base_type::impl_ref(Index<3>(idx0, idx1, idx2)); }

  const_reference_type impl_ref(index_type idx0, index_type idx1,
				  index_type idx2)
    const VSIP_NOTHROW
  { return base_type::impl_ref(Index<3>(idx0, idx1, idx2)); }
};

template <typename T, typename O, typename M>
class Dense<1, T, O, M> : public impl::Distributed_block<Dense<1, T, O>, M>
{
  typedef impl::Distributed_block<Dense<1, T, O>, M> base_type;
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

  // Internal user storage constructor.
  Dense(Domain<1> const&             dom,
	impl::User_storage<T> const& data,
	map_type const&              map = map_type())
    VSIP_THROW((std::bad_alloc))
      : base_type(dom, data, map)
    {}
};

/// Partial specialization of Dense class template for 1,2-dimension.
template <typename T, typename O, typename M>
class Dense<2, T, O, M> : public impl::Distributed_block<Dense<2, T, O>, M>
{
  typedef impl::Distributed_block<Dense<2, T, O>, M> base_type;
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

  reference_type impl_ref(index_type idx0, index_type idx1) VSIP_NOTHROW
  { return base_type::impl_ref(Index<2>(idx0, idx1));}

  const_reference_type impl_ref(index_type idx0, index_type idx1) const VSIP_NOTHROW
  { return base_type::impl_ref(Index<2>(idx0, idx1));}
};

template <typename T, typename O, typename M>
class Dense<3, T, O, M> : public impl::Distributed_block<Dense<3, T, O>, M>
{
  typedef impl::Distributed_block<Dense<3, T, O>, M> base_type;
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

  reference_type impl_ref(index_type idx0, index_type idx1, index_type idx2)
    VSIP_NOTHROW
  { return base_type::impl_ref(Index<3>(idx0, idx1, idx2)); }

  const_reference_type impl_ref(index_type idx0, index_type idx1,
				  index_type idx2)
    const VSIP_NOTHROW
  { return base_type::impl_ref(Index<3>(idx0, idx1, idx2)); }
};

/// Specialize block layout trait for Dense blocks.
template <dimension_type D, typename T, typename O, typename M>
struct get_block_layout<Dense<D, T, O, M> >
  : get_block_layout<impl::Strided<D, T,
      Layout<D, O, dense, impl::dense_complex_format>, M> >
{};

template <dimension_type D, typename T, typename O, typename M>
struct supports_dda<Dense<D, T, O, M> >
{ static bool const value = true;};

namespace impl
{

/// Specialize lvalue accessor trait for Dense blocks.
/// Dense provides direct lvalue accessors via impl_ref.
template <typename BlockT,
	  bool     use_proxy = is_split_block<BlockT>::value>
struct Dense_lvalue_factory_type;

template <typename BlockT>
struct Dense_lvalue_factory_type<BlockT, false>
{
  typedef True_lvalue_factory<BlockT> type;
  template <typename OtherBlock>
  struct Rebind {
    typedef True_lvalue_factory<OtherBlock> type;
  };
};

template <typename BlockT>
struct Dense_lvalue_factory_type<BlockT, true>
{
  typedef Proxy_lvalue_factory<BlockT> type;
  template <typename OtherBlock>
  struct Rebind {
    typedef Proxy_lvalue_factory<OtherBlock> type;
  };
};

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
struct Lvalue_factory_type<Dense<Dim, T, OrderT, Local_map> >
  : public Dense_lvalue_factory_type<Dense<Dim, T, OrderT, Local_map> >
{};



/// Specialize Distributed_local_block traits class for Dense.
///
/// For a serial map, distributed block and local block are the same.
template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
struct Distributed_local_block<Dense<Dim, T, OrderT, Local_map> >
{
  typedef Dense<Dim, T, OrderT, Local_map> type;
  typedef Dense<Dim, T, OrderT, Local_map> proxy_type;
};



/// For a distributed map, local block has a local map.
template <dimension_type D, typename T, typename O, typename M>
struct Distributed_local_block<Dense<D, T, O, M> >
{
  typedef Dense<D, T, O, Local_map> type;
  typedef typename Dense<D, T, O, M>::proxy_local_block_type proxy_type;
};



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



/// Assert that subblock is local to block (overload).
template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
void
assert_local(
  Dense<Dim, T, OrderT, Local_map> const& /*block*/,
  index_type                              sb)
{
  assert(sb == 0);
}



/// Assert that subblock is local to block (overload).
template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
void
assert_local(
  Dense<Dim, T, OrderT, MapT> const& block,
  index_type                         sb)
{
  block.assert_local(sb);
}



/// Specialize Is_simple_distributed_block traits class for Dense.
template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
struct Is_simple_distributed_block<Dense<Dim, T, OrderT, MapT> >
{
  static bool const value = true;
};



#if VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES==0

/// Specialize Combine_return_type for Dense block leaves.
template <typename       CombineT,
	  dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
struct Combine_return_type<CombineT, Dense<Dim, T, OrderT, MapT> >
{
  typedef Dense<Dim, T, OrderT, MapT> block_type;
  typedef typename CombineT::template return_type<block_type>::type
		type;
  typedef typename CombineT::template tree_type<block_type>::type
		tree_type;
};



/// Specialize apply_combine for Dense block leaves.
template <typename       CombineT,
	  dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
typename Combine_return_type<CombineT, Dense<Dim, T, OrderT, MapT> >::type
apply_combine(
  CombineT const&                    combine,
  Dense<Dim, T, OrderT, MapT> const& block)
{
  return combine.apply(block);
}
#endif



template <dimension_type Dim,
	  typename       T,
	  typename       OrderT>
struct Is_pas_block<Dense<Dim, T, OrderT, Local_map> >
{
  static bool const value = false;
};

template <dimension_type D,
	  typename       T,
	  typename       O,
	  typename       M>
struct Is_pas_block<Dense<D, T, O, M> >
  : Is_pas_block<Strided<D, T, Layout<D, O, dense, dense_complex_format>, M> >
{};



template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
struct is_modifiable_block<Dense<Dim, T, OrderT, MapT> >
{
  static bool const value = true;
};

#ifdef VSIP_IMPL_HAVE_CUDA
namespace cuda
{
template <dimension_type D, typename T, typename O>
struct has_device_storage<Dense<D, T, O> > { static bool const value = true;};
}
#endif

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_DENSE_HPP
