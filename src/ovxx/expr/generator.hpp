//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_generator_hpp_
#define ovxx_expr_generator_hpp_

#include <ovxx/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/length.hpp>

namespace ovxx
{
namespace parallel
{
template <dimension_type D>
class local_or_global_map;
} // namespace ovxx::parallel
namespace expr
{

/// Generator expression block type.
///
/// Template parameters:
///   D: to be a dimension with range 0 < D <= VSIP_MAX_DIMENSION
///   G: to be a functor class
template <dimension_type D, typename G>
class Generator : public G, ovxx::detail::nonassignable
{
public:
  static dimension_type const dim = D;
  typedef typename G::result_type value_type;

  typedef value_type&         reference_type;
  typedef value_type const&   const_reference_type;
  typedef parallel::local_or_global_map<D> map_type;

  Generator(Length<D> size) : size_(size) {}
  Generator(Length<D> size, G const &op) : G(op), size_(size) {}

  length_type size() const VSIP_NOTHROW { return total_size(size_);}
  length_type size(dimension_type block_dim OVXX_UNUSED, dimension_type d)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(block_dim == D); return size_[d];}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW { return map_;}

  value_type get(index_type i) const { return (*this)(i);}
  value_type get(index_type i, index_type j) const
  { return (*this)(i, j);}
  value_type get(index_type i, index_type j, index_type k) const
  { return (*this)(i, j, k);}

  // copy-constructor: default is OK.

private:
  Length<D> size_;
  map_type map_;
};

template <dimension_type D, typename G>
Generator<D, G> const&
get_local_block(Generator<D, G> const &block) { return block;}

template <dimension_type D, typename G>
void assert_local(Generator<D, G> const &, index_type) {}

} // namespace ovxx::expr

template <dimension_type D, typename G>
struct is_expr_block<expr::Generator<D, G> >
{ static bool const value = true;};

template <dimension_type D, typename G>
struct block_traits<expr::Generator<D, G> const>
  : by_value_traits<expr::Generator<D, G> const>
{};

template <dimension_type D, typename G>
struct block_traits<expr::Generator<D, G> >
{
  // No typedef provided.
};

template <dimension_type D, typename G>
struct distributed_local_block<expr::Generator<D, G> const>
{
  typedef expr::Generator<D, G> const type;
  typedef expr::Generator<D, G> const proxy_type;
};

namespace parallel
{
template <dimension_type D, typename G>
struct choose_peb<expr::Generator<D, G> const>
{ typedef peb_remap type;};

template <dimension_type D, typename G>
struct choose_peb<expr::Generator<D, G> >
{ typedef peb_remap type;};
} // namespace ovxx::parallel
} // namespace ovxx

#endif
