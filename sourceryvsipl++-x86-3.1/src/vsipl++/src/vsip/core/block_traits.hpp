/* Copyright (c) 2005 - 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_core_block_traits_hpp_
#define vsip_core_block_traits_hpp_

#ifndef VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES
#  define VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES 0
#endif

#include <vsip/core/refcount.hpp>
#include <vsip/core/layout.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip
{
namespace impl
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

/// Traits class to determine how a view will refer to its block.
///
/// By default, blocks should be refered to by reference, using the
/// Ref_counted_ptr class.
///
/// However, expression template blocks need to be held by-value,
/// as a data member of the view, using the Stored_value class.
template <typename Block>
struct By_ref_block_storage
{
  typedef Ref_counted_ptr<Block> type;
  typedef Block&                 plain_type;
  typedef Block const&           expr_type;

  template <typename RP>
  struct With_rp
  {
    typedef RPPtr<Block, RP> type;
  };
};

template <typename Block>
struct By_value_block_storage
{
  typedef Stored_value<Block> type;
  typedef Block               plain_type;
  typedef Block               expr_type;

  template <typename RP>
  struct With_rp
  {
    typedef Stored_value<Block> type;
  };
};

template <typename Block>
struct View_block_storage : By_ref_block_storage<Block>
{};



// Example: Defining storage type for Dense could be done as follows
// (although this is unnecessary since it uses the default type:
//  Ref_counted_ptr)
/*
template <dimension_type Dim,
	  typename       T,
	  typename       Order,
	  typename       Map>
struct View_block_storage<Dense<Dim, T, Order, Map> >
{
  typedef Ref_counted_ptr<Dense<Dim, T, Order, Map> > type;
};
*/


/// Traits class to determine how an expression block will refer to 
/// its operand blocks.
///
/// By default, blocks should be refered to by reference.
///
/// However, scalar blocks need to be held by-value,
/// as they are implicitely constructed by operators.
template <typename Block>
struct Expr_block_storage
{
  typedef Block const& type;
};


/// Traits class to determine the local block used for distributed blocks. 
///
/// The primary definition works for non-distributed blocks were the
/// local block is just the block type.
///
/// :type: indicates the local block type.
/// :proxy_type: indicates the proxy local block type, to be used for
///              querying layout of local blocks on remote processors.
template <typename Block>
struct Distributed_local_block
{
  typedef Block type;
  typedef Block proxy_type;
};



/// Traits class to determine the root block for a stack of subblocks.
template <typename BlockT>
struct Block_root
{
  typedef BlockT type;
};

template <typename BlockT>
BlockT const&
block_root(BlockT const& block)
{
  return block;
}

template <typename BlockT>
struct is_split_block
{
  static bool const value =
    is_complex<typename BlockT::value_type>::value &&
    get_block_layout<BlockT>::storage_format == split_complex;
};



/// Traits class to determine if block is a simple distributed block.
template <typename Block>
struct Is_simple_distributed_block { static bool const value = false; };



/// Traits class to determine if block is an expression block.
template <typename Block>
struct is_expr_block
{ static bool const value = false; };

template <typename Block>
struct is_expr_block<const Block>
{ static bool const value = is_expr_block<Block>::value; };



/// Temporary trait to determine "proper" type of block.
///
/// Dispatch templates can sometimes strip the 'const'
/// off of a block type.  This causes problems for exprssion
/// block, which *must* be const.
template <typename BlockT>
struct Proper_type_of
{
  typedef typename
          conditional<is_expr_block<BlockT>::value,
		      BlockT const,
		      typename remove_const<BlockT>::type>::type
          type;
};



/// Check that block type is proper.
template <typename BlockT,
	  bool     valid = is_same<BlockT,
				typename Proper_type_of<BlockT>::type>::value>
struct Assert_proper_block;

template <typename BlockT>
struct Assert_proper_block<BlockT, true> {};



/// Traits class to determine if block is a scalar block.
template <typename Block>
struct is_scalar_block
{ static bool const value = false; };

template <typename Block>
struct is_scalar_block<const Block>
{ static bool const value = is_scalar_block<Block>::value; };



/// Traits class to determine if block has a size.
template <typename Block>
struct Is_sized_block
{ static bool const value = true; };

template <typename Block>
struct Is_sized_block<const Block>
{ static bool const value = Is_sized_block<Block>::value; };



/// Traits class to determine if a block is a leaf block in an
/// expressions.
template <typename BlockT>
struct Is_leaf_block
{
  static bool const value =
    !is_expr_block<BlockT>::value ||
     is_scalar_block<BlockT>::value;
};



template <dimension_type Dim,
	  typename       Map1,
	  typename       Map2>
struct Map_equal
{
  static bool value(Map1 const&, Map2 const&)
    { return false; }
};

template <dimension_type Dim,
	  typename       Map1,
	  typename       Map2>
inline bool
map_equal(Map1 const& map1, Map2 const& map2)
{
  return Map_equal<Dim, Map1, Map2>::value(map1, map2);
}



/// Check if lhs map is same as rhs block's map.
///
/// Two blocks are the same if they distribute a view into subblocks
/// containing the same elements.
template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT>
struct Is_par_same_map
{
  static bool value(MapT const& map, BlockT const& block)
    { return map_equal<Dim>(map, block.map()); }
};



template <typename BlockT>
struct Is_par_reorg_ok
{
  static bool const value = true;
};



/// @group Implementation tags for Par_expr_block: {

/// Reorganize block
struct Peb_reorg_tag;
/// Reuse block directly
struct Peb_reuse_tag;
/// Reuse block, but with different mapping
struct Peb_remap_tag;

/// }

/// Traits class to choose the appropriate Par_expr_block impl tag for
/// a block type. By default, blocks should be reorganized.
template <typename BlockT>
struct Choose_peb { typedef Peb_reorg_tag type; };



// Forward Declaration
template <dimension_type Dim,
	  typename       MapT,
	  typename       BlockT,
	  typename       ImplTag = typename Choose_peb<BlockT>::type>
class Par_expr_block;



template <typename CombineT,
	  typename BlockT>
struct Combine_return_type;



#if VSIP_IMPL_USE_GENERIC_VISITOR_TEMPLATES==1

/// General case for return type of expression tree transformations.
///
/// This general case covers all leaf blocks in expression trees.
/// However, to aid in debugging, it is useful to avoid providing a
/// general case and only provide specializations.  Missing required
/// specializations then result in compilation errors.
///
/// Currently leaf specializations are provided by Dense,
/// Distributed_block, and Par_expr_block.
template <typename CombineT,
	  typename BlockT>
struct Combine_return_type
{
  typedef typename CombineT::template return_type<BlockT>::type
		type;
  typedef typename CombineT::template tree_type<BlockT>::type
		tree_type;
};



template <typename CombineT,
	  typename BlockT>
typename Combine_return_type<CombineT, BlockT>::type
apply_combine(
  CombineT const& combine,
  BlockT const&   block)
{
  return combine.apply(block);
}




template <typename VisitorT,
	  typename BlockT>
void
apply_leaf(
  VisitorT const& visitor,
  BlockT const&   block)
{
  visitor.apply(block);
}

#endif

/// Block lvalue accessor factories.

template <typename Block> class Proxy_lvalue_factory;
template <typename Block> class True_lvalue_factory;

/// Traits class to determine whether a block provides an lvalue accessor.
/// The ::type member of this class, when instantiated for a block type,
/// will be one of the above factory classes.  By default, we assume there
/// is no direct access to lvalues, so we go through a proxy class that calls
/// get() and put() [see lvalue-proxy.hpp].
/// The Rebind nested class is for use in specializations that want to say
/// "make the same choice that that block makes"; see subblock.hpp for examples.
template <typename Block>
struct Lvalue_factory_type
{
  typedef Proxy_lvalue_factory<Block> type;
  template <typename OtherBlock>
  struct Rebind {
    typedef Proxy_lvalue_factory<OtherBlock> type;
  };
};



/// Traits class to determine if a block has a valid PAS distribution
/// handle (which allows collective assignment to be used).  Blocks
/// without a valid PAS distribution must use the low-level direct
/// assignment.
template <typename BlockT>
struct Is_pas_block
{
  static bool const value = false;
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
struct Has_put
{
  static bool const value = 
  sizeof(has_put_helper<BlockT>(0)) == sizeof(yes_tag);
};

} // namespace detail

template <typename BlockT>
struct is_modifiable_block
{
  static bool const value = detail::Has_put<BlockT>::value;
};

// const blocks are not modifiable.
template <typename B>
struct is_modifiable_block<B const>
{
  static bool const value = false;
};

template <typename Block, bool valid = is_modifiable_block<Block>::value>
struct Assert_modifiable_block;

template <typename Block>
struct Assert_modifiable_block<Block, true> {};


// Compare two blocks for equality.

template <typename Block1,
	  typename Block2>
struct Is_same_block
{
  static bool compare(Block1 const&, Block2 const&) { return false; }
};

template <typename BlockT>
struct Is_same_block<BlockT, BlockT>
{
  static bool compare(BlockT const& a, BlockT const& b) { return &a == &b; }
};

template <typename Block1,
	  typename Block2>
bool
is_same_block(
  Block1 const& a,
  Block2 const& b)
{
  return Is_same_block<Block1, Block2>::compare(a, b);
}

/// If provided type is complex, extract the component type,
/// otherwise use the provided bogus type.
template <typename T, typename BogusT>
struct Complex_value_type
{
  typedef BogusT type; 
  typedef BogusT ptr_type;
};

template <typename T, typename BogusT>
struct Complex_value_type<complex<T>, BogusT>
{
  typedef T  type; 
  typedef T *ptr_type;
};

} // namespace impl
} // namespace vsip

#endif // VSIP_CORE_BLOCK_TRAITS_HPP 
