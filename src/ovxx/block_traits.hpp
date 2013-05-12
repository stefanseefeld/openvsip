//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_block_traits_hpp_
#define ovxx_block_traits_hpp_

#include <ovxx/support.hpp>
#include <ovxx/refcounted.hpp>
#include <ovxx/detail/util.hpp>
#include <ovxx/complex_traits.hpp>
#include <ovxx/layout.hpp>

namespace ovxx
{
/// Check whether the given block argument is allocatable.
template <typename B>
class is_allocatable_block
{
  template <int x>
  class helper{};

  // Function template that matches if 'U' supports a particular constructor.
  template <typename U>
  static detail::yes_tag 
  sfinae(helper<sizeof U(Domain<U::dim>(), typename U::map_type())> *);

  // Fallback function template.
  template<typename U>
  static detail::no_tag sfinae(...);

public:
  static bool const value = sizeof(sfinae<B>(0)) == sizeof(detail::yes_tag);
};

namespace detail
{

template <typename T, void (T::*)(index_type, typename T::value_type)>
struct ptmf_helper;

template <typename T>
no_tag
has_put_helper(...);

template <typename T>
yes_tag
has_put_helper(int, ptmf_helper<T, &T::put>* p = 0);

template <typename BlockT>
struct has_put
{
  static bool const value = 
  sizeof(has_put_helper<BlockT>(0)) == sizeof(yes_tag);
};

} // namespace detail

template <typename B>
struct is_modifiable_block
{
  static bool const value = detail::has_put<B>::value;
};

// const blocks are not modifiable.
template <typename B>
struct is_modifiable_block<B const>
{
  static bool const value = false;
};

template <typename B>
struct is_split_block
{
  static bool const value =
    is_complex<typename B::value_type>::value &&
    get_block_layout<B>::storage_format == split_complex;
};

/// Traits class to determine if block is an expression block.
template <typename B>
struct is_expr_block { static bool const value = false;};

template <typename B>
struct is_expr_block<B const>
{ static bool const value = is_expr_block<B>::value;};

/// Traits class to determine the local block used for distributed blocks. 
///
/// The primary definition works for non-distributed blocks where the
/// local block is just the block type.
///
/// :type: indicates the local block type.
/// :proxy_type: indicates the proxy local block type, to be used for
///              querying layout of local blocks on remote processors.
template <typename B>
struct distributed_local_block
{
  typedef B type;
  typedef B proxy_type;
};

/// Traits class to determine if block is a simple distributed block.
template <typename B>
struct is_simple_distributed_block { static bool const value = false;};

/// Traits class to determine if block is a scalar block.
template <typename B>
struct is_scalar_block { static bool const value = false;};

template <typename B>
struct is_scalar_block<B const>
{ static bool const value = is_scalar_block<B>::value;};

/// Traits class to determine if block has a size.
template <typename B>
struct is_sized_block { static bool const value = true;};

template <typename B>
struct is_sized_block<B const>
{ static bool const value = is_sized_block<B>::value;};

/// Traits class to determine if a block is a leaf block in an
/// expression.
template <typename B>
struct is_leaf_block
{
  static bool const value = !is_expr_block<B>::value || is_scalar_block<B>::value;
};

namespace detail
{

// Interface a value-type as if it was a smart-ptr.
template <typename T>
class by_value_proxy : noncopyable
{
public:
  explicit by_value_proxy(T *p) : data_(*p) {}
  by_value_proxy(T *p, bool) : data_(*p) {}
  ~by_value_proxy() {}

  T &operator*() const { return data_;}
  T *operator->() const { return &data_;}
  T *get() const { return &data_;}

private:
  mutable T data_;
};

template <typename T>
class by_value_proxy<T const> : noncopyable
{
public:
  explicit by_value_proxy(T const *p) : data_(*p) {}
  by_value_proxy(T const *p, bool) : data_(*p) {}
  ~by_value_proxy() {}

  T const &operator*() const { return data_;}
  T const *operator->() const { return &data_;}
  T const *get() const { return &data_;}

private:
  T const data_;
};

} // namespace ovxx::detail

template <typename B>
struct by_ref_traits
{
  typedef refcounted_ptr<B> ptr_type;
  typedef B &plain_type;
  typedef B const &expr_type;
};

template <typename B>
struct by_value_traits
{
  typedef detail::by_value_proxy<B> ptr_type;
  typedef B plain_type;
  typedef B expr_type;
};

/// By default, blocks are ref-counted.
/// However, for certain block types we can do better...
template <typename B>
struct block_traits : by_ref_traits<B> {};

template <typename B>
struct is_par_reorg_ok
{
  static bool const value = true;
};

template <typename B>
inline typename B::value_type
get(B const &block, Index<1> const &idx)
{
  return block.get(idx[0]);
}

template <typename B>
inline typename B::value_type
get(B const &block, Index<2> const &idx)
{
  return block.get(idx[0], idx[1]);
}

template <typename B>
inline typename B::value_type
get(B const &block, Index<3> const &idx)
{
  return block.get(idx[0], idx[1], idx[2]);
}

template <typename B>
void
put(B &block, Index<1> const &idx, typename B::value_type value)
{
  block.put(idx[0], value);
}

template <typename B>
void
put(B &block, Index<2> const &idx, typename B::value_type value)
{
  block.put(idx[0], idx[1], value);
}

template <typename B>
void
put(B &block, Index<3> const &idx, typename B::value_type value)
{
  block.put(idx[0], idx[1], idx[2], value);
}

namespace parallel
{

/// @group Implementation tags for Expr_block: {

/// Reorganize block
struct peb_reorg;
/// Reuse block directly
struct peb_reuse;
/// Reuse block, but with different mapping
struct peb_remap;

/// }

/// Traits class to choose the appropriate Expr_block impl tag for
/// a block type. By default, blocks should be reorganized.
template <typename B>
struct choose_peb { typedef peb_reorg type;};

} // namespace ovxx::parallel

/// @group lvalue access to block elements {

template <typename Block> class proxy_factory;
template <typename Block> class ref_factory;

/// Traits class to determine whether a block provides an lvalue accessor.
/// The ::type member of this class, when instantiated for a block type,
/// will be one of the above factory classes.  By default, we assume there
/// is no direct access to lvalues, so we go through a proxy class that calls
/// get() and put() [see element_proxy.hpp].
/// The rebind nested class is for use in specializations that want to say
/// "make the same choice that that block makes"; see subblock.hpp for examples.
template <typename B>
struct lvalue_factory_type
{
  typedef proxy_factory<B> type;
  template <typename O>
  struct rebind { typedef proxy_factory<O> type;};
};

/// }

template <typename B1, typename B2>
bool is_same_block(B1 const &a, B2 const &b)
{
  return detail::is_same<B1, B2>::compare(a, b);
}

} // namespace ovxx

#endif
