//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_EXPR_SCALAR_BLOCK_HPP
#define VSIP_CORE_EXPR_SCALAR_BLOCK_HPP

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/map_fwd.hpp>

namespace vsip
{
namespace impl
{
template <dimension_type D>
struct Scalar_block_shared_map
{
  // typedef Local_or_global_map<D> type;
  typedef Scalar_block_map<D> type;
  static type map;
};

// These are explicitly instantiated in scalar_block.cpp
#if defined (__ghs__)
#pragma do_not_instantiate Scalar_block_shared_map<1>::map
#pragma do_not_instantiate Scalar_block_shared_map<2>::map
#pragma do_not_instantiate Scalar_block_shared_map<3>::map
#endif


/// An adapter presenting a scalar as a block. This is useful when constructing
/// binary expression blocks (which expect two block operands) taking a block and
/// a scalar.
///
/// Template parameters:
///   :D: to be a dimension with range 0 < D <= VSIP_MAX_DIMENSION
///   :Scalar: to be a builtin scalar type.
template <dimension_type D, typename Scalar>
class Scalar_block_base : public Non_assignable
{
public:
  typedef Scalar value_type;
  typedef value_type& reference_type;
  typedef value_type const& const_reference_type;
  typedef typename Scalar_block_shared_map<D>::type map_type;

  static dimension_type const dim = D;

  Scalar_block_base(Scalar s) : value_(s) {}
#if (defined(__GNUC__) && __GNUC__ < 4)
  // GCC 3.4.4 appears to over-optimize multiple scalar values on
  // stack when optimization & strong inlining are enabled, causing
  // threshold.cpp and other tests to fail.  (070618)
  Scalar_block_base(Scalar_block_base const& b) : value_(b.value_) {}
#endif

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW
  { return Scalar_block_shared_map<D>::map; }

  Scalar value() const VSIP_NOTHROW {return value_;}

private:
  Scalar const         value_;
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace expr
{

template <dimension_type D, typename T>
class Scalar;

/// Scalar specialization for 1-dimension.
template <typename T>
class Scalar<1, T> : public impl::Scalar_block_base<1, T>
{
public:
  Scalar(T s) : impl::Scalar_block_base<1, T>(s) {}
  length_type size() const VSIP_NOTHROW { return 0;}
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW { return 0;}
  T get(index_type) const VSIP_NOTHROW { return this->value();}
};

/// Scalar specialization for 2-dimension.
template <typename T>
class Scalar<2, T> : public impl::Scalar_block_base<2, T>
{
public:
  Scalar(T s) : impl::Scalar_block_base<2, T>(s) {}
  length_type size() const VSIP_NOTHROW { return 0;}
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW { return 0;}

  T get(index_type) const VSIP_NOTHROW { return this->value();}
  T get(index_type, index_type) const VSIP_NOTHROW { return this->value();}
};

/// Scalar specialization for 3-dimension.
template <typename T>
class Scalar<3, T> : public impl::Scalar_block_base<3, T>
{
public:
  Scalar(T s) : impl::Scalar_block_base<3, T>(s) {}

  length_type size() const VSIP_NOTHROW { return 0;}
  length_type size(dimension_type, dimension_type) const VSIP_NOTHROW { return 0;}
  T get(index_type) const VSIP_NOTHROW { return this->value();}
  T get(index_type, index_type, index_type) const VSIP_NOTHROW { return this->value();}
};

} // namespace vsip_csl::expr
} // namespace vsip_csl

namespace vsip
{
namespace impl
{
namespace expr = vsip_csl::expr;
template <dimension_type D, typename T>
struct is_expr_block<expr::Scalar<D, T> >
{ static bool const value = true;};

template <dimension_type D, typename T>
struct Is_sized_block<expr::Scalar<D, T> >
{ static bool const value = false;};

template <dimension_type D, typename T>
struct is_scalar_block<expr::Scalar<D, T> >
{ static bool const value = true;};

template <dimension_type D, typename T>
struct Distributed_local_block<expr::Scalar<D, T> const>
{
  typedef expr::Scalar<D, T> const type;
  typedef expr::Scalar<D, T> const proxy_type;
};

template <dimension_type D, typename T>
struct Distributed_local_block<expr::Scalar<D, T> >
{
  typedef expr::Scalar<D, T> type;
  typedef expr::Scalar<D, T> proxy_type;
};

template <dimension_type D, typename T>
expr::Scalar<D, T>
get_local_block(expr::Scalar<D, T> const &block) { return block;}

template <typename CombineT, dimension_type D, typename T>
struct Combine_return_type<CombineT, expr::Scalar<D, T> const>
{
  typedef expr::Scalar<D, T> block_type;
  typedef typename CombineT::template return_type<block_type>::type
  type;
  typedef typename CombineT::template tree_type<block_type>::type
  tree_type;
};

template <typename CombineT, dimension_type D, typename T>
struct Combine_return_type<CombineT, expr::Scalar<D, T> >
  : Combine_return_type<CombineT, expr::Scalar<D, T> const>
{};

template <typename CombineT, dimension_type D, typename T>
typename Combine_return_type<CombineT,
			     expr::Scalar<D, T> const>::type
apply_combine(CombineT const &combine, expr::Scalar<D, T> const &block)
{
  return combine.apply(block);
}

template <typename VisitorT, dimension_type D, typename T>
void
apply_leaf(VisitorT const &visitor, expr::Scalar<D, T> const &block)
{
  visitor.apply(block);
}

template <dimension_type MD, typename M, dimension_type D, typename T>
struct Is_par_same_map<MD, M, expr::Scalar<D, T> >
{
  typedef expr::Scalar<D, T> block_type;
  static bool value(M const&, block_type const&) { return true;}
};

template <dimension_type MD, typename M, dimension_type D, typename T>
struct Is_par_same_map<MD, M, expr::Scalar<D, T> const>
{
  typedef expr::Scalar<D, T> block_type;
  static bool value(M const&, block_type const&) { return true;}
};


// Default Is_par_reorg_ok is OK.



/// Assert that subblock is local to block (overload).
template <dimension_type D, typename T>
void
assert_local(expr::Scalar<D, T> const &block, index_type sb)
{
  // Scalar_block is always valid locally.
}

template <dimension_type D, typename T>
struct Choose_peb<expr::Scalar<D, T> >
{ typedef Peb_reuse_tag type; };

/// Store Scalar_blocks by-value.
template <dimension_type D, typename T>
struct View_block_storage<expr::Scalar<D, T> const>
  : By_value_block_storage<expr::Scalar<D, T> const>
{};

template <dimension_type D, typename T>
struct View_block_storage<expr::Scalar<D, T> >
  : By_value_block_storage<expr::Scalar<D, T> >
{};

template <dimension_type D, typename T>
struct Expr_block_storage<expr::Scalar<D, T> >
{
  typedef expr::Scalar<D, T> type;
};

} // namespace vsip::impl
} // namespace vsip

#endif
