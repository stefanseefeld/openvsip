/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/generator_block.hpp
    @author  Jules Bergmann
    @date    2005-08-15
    @brief   VSIPL++ Library: "Generator" expression block class templates.
*/

#ifndef VSIP_CORE_EXPR_GENERATOR_BLOCK_HPP
#define VSIP_CORE_EXPR_GENERATOR_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/parallel/local_map.hpp>

namespace vsip
{
namespace impl
{

/***********************************************************************
  Declarations
***********************************************************************/

/// Expression template block for Generator expressions.
///
/// Requires:
///   DIM to be a dimension with range 0 < D <= VSIP_MAX_DIMENSION
///   GENERATOR to be a functor class with the following members:
///      OPERATOR()() to compute a value based given indices.
///      RESULT_TYPE to be the result type of operator()()

template <dimension_type Dim,
	  typename       Generator>
class Generator_expr_block
  : public Generator,
    public Non_assignable
{
  // Compile-time values and typedefs.
public:
  static dimension_type const dim = Dim;
  typedef typename Generator::result_type value_type;

  typedef value_type&         reference_type;
  typedef value_type const&   const_reference_type;
  typedef Local_or_global_map<Dim> map_type;


  // Constructors.
public:
  Generator_expr_block(Length<Dim> size)
    : size_(size) {}
  Generator_expr_block(Length<Dim> size, Generator const& op)
    : Generator(op), size_(size) {}


  // Accessors.
public:
  length_type size() const VSIP_NOTHROW
  { return total_size(size_); }

  length_type size(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  { assert(block_dim == Dim); return size_[d]; }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW { return map_;}

  value_type get(index_type i) const;
  value_type get(index_type i, index_type j) const;
  value_type get(index_type i, index_type j, index_type k) const;

  // copy-constructor: default is OK.

  // Member data.
private:
  Length<Dim> size_;
  map_type    map_;
};



/// Specialize Is_expr_block for generator expr blocks.

template <dimension_type Dim,
	  typename       Generator>
struct Is_expr_block<Generator_expr_block<Dim, Generator> >
{ static bool const value = true; };



/// Specialize View_block_storage to control how views store generator
/// expression template blocks.

template <dimension_type Dim,
	  typename       Generator>
struct View_block_storage<const Generator_expr_block<Dim, Generator> >
  : By_value_block_storage<const Generator_expr_block<Dim, Generator> >
{};



template <dimension_type Dim,
	  typename       Generator>
struct View_block_storage<Generator_expr_block<Dim, Generator> >
{
  // No typedef provided.
};



/***********************************************************************
  Distributed Traits
***********************************************************************/

// NOTE: Distributed_local_block needs to be defined for const
// Generator_expr_block, not regular Generator_expr_block.

template <dimension_type Dim,
	  typename       Generator>
struct Distributed_local_block<Generator_expr_block<Dim, Generator> const>
{
  typedef Generator_expr_block<Dim, Generator> const type;
  typedef Generator_expr_block<Dim, Generator> const proxy_type;
};



template <dimension_type Dim,
	  typename       Generator>
Generator_expr_block<Dim, Generator> const&
get_local_block(
  Generator_expr_block<Dim, Generator> const& block)
{
  return block;
}



template <dimension_type Dim,
	  typename       Generator>
void
assert_local(
  Generator_expr_block<Dim, Generator> const& /*block*/,
  index_type                                  /*sb*/)
{
}


template <dimension_type Dim, typename Generator>
struct Choose_peb<Generator_expr_block<Dim, Generator> const>
{ typedef Peb_remap_tag type; };

template <dimension_type Dim, typename Generator>
struct Choose_peb<Generator_expr_block<Dim, Generator> >
{ typedef Peb_remap_tag type; };


template <typename       CombineT,
	  dimension_type Dim,
	  typename       Generator>
struct Combine_return_type<CombineT,
			   Generator_expr_block<Dim, Generator> const>
{
  typedef Generator_expr_block<Dim, Generator> block_type;
  typedef typename CombineT::template return_type<block_type>::type
		type;
  typedef typename CombineT::template tree_type<block_type>::type
		tree_type;
};



template <typename       CombineT,
	  dimension_type Dim,
	  typename       Generator>
struct Combine_return_type<CombineT, Generator_expr_block<Dim, Generator> >
{
  typedef Generator_expr_block<Dim, Generator> block_type;
  typedef typename CombineT::template return_type<block_type>::type
		type;
  typedef typename CombineT::template tree_type<block_type>::type
		tree_type;
};



template <typename       CombineT,
	  dimension_type Dim,
	  typename       Generator>
typename Combine_return_type<CombineT,
			     Generator_expr_block<Dim, Generator> const>::type
apply_combine(
  CombineT const&                             combine,
  Generator_expr_block<Dim, Generator> const& block)
{
  return combine.apply(block);
}



template <typename       VisitorT,
	  dimension_type Dim,
	  typename       Generator>
void
apply_leaf(
  VisitorT const&                             /*visitor*/,
  Generator_expr_block<Dim, Generator> const& /*block*/)
{
  // No-op
}


// Is_par_same_map primary case works for Generator_expr_block.



/***********************************************************************
  Definitions
***********************************************************************/

template <dimension_type Dim,
	  typename       Generator>
inline typename Generator_expr_block<Dim, Generator>::value_type
Generator_expr_block<Dim, Generator>::get(index_type i) const
{
  return (*this)(i);
}

template <dimension_type Dim,
	  typename       Generator>
inline typename Generator_expr_block<Dim, Generator>::value_type
Generator_expr_block<Dim, Generator>::get(
  index_type i,
  index_type j) const
{
  return (*this)(i, j);
}

template <dimension_type Dim,
	  typename       Generator>
inline typename Generator_expr_block<Dim, Generator>::value_type
Generator_expr_block<Dim, Generator>::get(
  index_type i,
  index_type j,
  index_type k) const
{
  return (*this)(i, j, k);
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_EXPR_GENERATOR_BLOCK_HPP
