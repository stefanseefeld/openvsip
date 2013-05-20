//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_scalar_hpp_
#define ovxx_expr_scalar_hpp_

#include <ovxx/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <vsip/impl/map_fwd.hpp>

namespace ovxx
{
namespace expr
{
namespace detail
{
template <dimension_type D>
struct shared_scalar_map
{
  typedef parallel::scalar_map<D> type;
  static type map;
};

template <dimension_type D> 
parallel::scalar_map<D> shared_scalar_map<D>::map;


/// An adapter presenting a scalar as a block. This is useful when constructing
/// binary expression blocks (which expect two block operands) taking a block and
/// a scalar.
///
/// Template parameters:
///   :D: to be a dimension with range 0 < D <= VSIP_MAX_DIMENSION
///   :Scalar: to be a builtin scalar type.
template <dimension_type D, typename Scalar>
class Scalar_block_base : ovxx::detail::nonassignable
{
public:
  typedef Scalar value_type;
  typedef value_type& reference_type;
  typedef value_type const& const_reference_type;
  typedef parallel::scalar_map<D> map_type;

  static dimension_type const dim = D;

  Scalar_block_base(Scalar s) : value_(s) {}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW
  { return shared_scalar_map<D>::map;}

  Scalar value() const VSIP_NOTHROW {return value_;}

private:
  Scalar const         value_;
};

} // namespace ovxx::expr::detail

template <dimension_type D, typename T>
class Scalar;

/// Scalar specialization for 1-dimension.
template <typename T>
class Scalar<1, T> : public detail::Scalar_block_base<1, T>
{
public:
  Scalar(T s) : detail::Scalar_block_base<1, T>(s) {}
  length_type size() const VSIP_NOTHROW { return 0;}
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW { return 0;}
  T get(index_type) const VSIP_NOTHROW { return this->value();}
};

/// Scalar specialization for 2-dimension.
template <typename T>
class Scalar<2, T> : public detail::Scalar_block_base<2, T>
{
public:
  Scalar(T s) : detail::Scalar_block_base<2, T>(s) {}
  length_type size() const VSIP_NOTHROW { return 0;}
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW { return 0;}

  T get(index_type) const VSIP_NOTHROW { return this->value();}
  T get(index_type, index_type) const VSIP_NOTHROW { return this->value();}
};

/// Scalar specialization for 3-dimension.
template <typename T>
class Scalar<3, T> : public detail::Scalar_block_base<3, T>
{
public:
  Scalar(T s) : detail::Scalar_block_base<3, T>(s) {}

  length_type size() const VSIP_NOTHROW { return 0;}
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW { return 0;}
  T get(index_type) const VSIP_NOTHROW { return this->value();}
  T get(index_type, index_type, index_type) const VSIP_NOTHROW { return this->value();}
};

template <dimension_type D, typename T>
Scalar<D, T> get_local_block(Scalar<D, T> const &block) { return block;}

} // namespace ovxx::expr

template <dimension_type D, typename T>
struct is_expr_block<expr::Scalar<D, T> >
{ static bool const value = true;};

template <dimension_type D, typename T>
struct is_sized_block<expr::Scalar<D, T> >
{ static bool const value = false;};

template <dimension_type D, typename T>
struct is_scalar_block<expr::Scalar<D, T> >
{ static bool const value = true;};

template <dimension_type D, typename T>
struct block_traits<expr::Scalar<D, T> const>
  : by_value_traits<expr::Scalar<D, T> const>
{};

template <dimension_type D, typename T>
struct block_traits<expr::Scalar<D, T> >
  : by_value_traits<expr::Scalar<D, T> >
{};

template <dimension_type D, typename T>
struct distributed_local_block<expr::Scalar<D, T> const>
{
  typedef expr::Scalar<D, T> const type;
  typedef expr::Scalar<D, T> const proxy_type;
};

template <dimension_type D, typename T>
struct distributed_local_block<expr::Scalar<D, T> >
{
  typedef expr::Scalar<D, T> type;
  typedef expr::Scalar<D, T> proxy_type;
};

namespace parallel
{
template <dimension_type D, typename T>
struct choose_peb<expr::Scalar<D, T> >
{ typedef peb_reuse type;};

} // namespace ovxx::parallel

template <dimension_type D, typename T>
void assert_local(expr::Scalar<D, T> const &, index_type) {}

} // namespace ovxx

#endif
