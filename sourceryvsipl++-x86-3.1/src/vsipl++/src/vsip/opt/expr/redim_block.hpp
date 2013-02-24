/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_EXPR_REDIM_BLOCK_HPP
#define VSIP_OPT_EXPR_REDIM_BLOCK_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/storage.hpp>
#include <vsip/dda.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>

namespace vsip
{
namespace impl
{

/// redim_get and redim_put are helper functions for Redim_block
/// They allow a single Redim_block class definition to reduce both
/// 2-dimensional and 3-dimensional blocks.
template <typename BlockT>
typename BlockT::value_type
redim_get(BlockT const& blk, index_type l_idx, integral_constant<int, 2>)
{
  typedef typename get_block_layout<BlockT>::order_type OrderT;

  index_type idx[2];

  idx[OrderT::impl_dim1] = l_idx % blk.size(2, OrderT::impl_dim1);
  idx[OrderT::impl_dim0] = l_idx / blk.size(2, OrderT::impl_dim1);

  return blk.get(idx[0], idx[1]);
}

template <typename BlockT>
void
redim_put(BlockT &blk,
	  index_type l_idx,
	  typename BlockT::value_type value,
	  integral_constant<int, 2>)
{
  typedef typename get_block_layout<BlockT>::order_type OrderT;

  index_type idx[2];

  idx[OrderT::impl_dim1] = l_idx % blk.size(2, OrderT::impl_dim1);
  idx[OrderT::impl_dim0] = l_idx / blk.size(2, OrderT::impl_dim1);

  blk.put(idx[0], idx[1], value);
}

template <typename BlockT>
typename BlockT::value_type
redim_get(BlockT const& blk, index_type l_idx, integral_constant<int, 3>)
{
  typedef typename get_block_layout<BlockT>::order_type OrderT;

  index_type idx[3];

  idx[OrderT::impl_dim2] = l_idx % blk.size(3, OrderT::impl_dim2);
  l_idx /= blk.size(3, OrderT::impl_dim2);
  idx[OrderT::impl_dim1] = l_idx % blk.size(3, OrderT::impl_dim1);
  idx[OrderT::impl_dim0] = l_idx / blk.size(3, OrderT::impl_dim1);

  return blk.get(idx[0], idx[1], idx[2]);
}

template <typename BlockT>
void
redim_put(BlockT &blk,
	  index_type l_idx,
	  typename BlockT::value_type value,
	  integral_constant<int, 3>)
{
  typedef typename get_block_layout<BlockT>::order_type OrderT;

  index_type idx[3];

  idx[OrderT::impl_dim2] = l_idx % blk.size(3, OrderT::impl_dim2);
  l_idx /= blk.size(3, OrderT::impl_dim2);
  idx[OrderT::impl_dim1] = l_idx % blk.size(3, OrderT::impl_dim1);
  idx[OrderT::impl_dim0] = l_idx / blk.size(3, OrderT::impl_dim1);

  blk.put(idx[0], idx[1], idx[2], value);
}

// The following version of Redim_block appears to cause an ICE with
// various versions of G++, notably ppu-g++ 4.1.1, so we fall back
// to a version that can't distinguish between modifiable and non-modifiable
// versions (and thus needs to const-cast the data-type.
#if 0

/// Redimension block.
///
/// Provides a 1-dimensional view of a multidimensional block.
/// Intended for use when a multidimensional block refers to dense
/// data, but does not support 1,x-dimensional access (for example
/// a Sliced_block).  Redim_block's direct data interface requires
/// underlying block to be dense, but get/put work regardless of the
/// layout.
template <typename Block,
 	  dimension_type D, 
 	  bool Mutable = is_modifiable_block<Block>::value>
class Redim_block;

template <typename B, dimension_type D>
class Redim_block<B, D, false> : Compile_time_assert<D == 2 || D == 3>
{
public:
  static dimension_type const dim = 1;

  typedef typename B::value_type value_type;
  typedef typename B::reference_type reference_type;
  typedef typename B::const_reference_type const_reference_type;
  typedef typename B::map_type map_type;

  typedef typename get_block_layout<B>::order_type raw_order_type;

  typedef typename B::const_ptr_type ptr_type;

  Redim_block(B &block) : blk_(&block) {}
  Redim_block(Redim_block const &rb) VSIP_NOTHROW : blk_(&*rb.blk_) {}
  ~Redim_block() VSIP_NOTHROW {}

  value_type get(index_type idx) const VSIP_NOTHROW
  { return redim_get(*blk_, idx, integral_constant<int, D>());}

  length_type size() const VSIP_NOTHROW
  { return blk_->size();}

  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  {
    assert(block_dim == 1 && d == 0);
    return blk_->size();
  }

  map_type const& map() const VSIP_NOTHROW
  { return blk_->map_;}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  ptr_type ptr() const VSIP_NOTHROW { return blk_->ptr();}

  stride_type stride(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    assert(block_dim == 1 && d == 0);
    return D == 2 ?
      blk_->stride(2, raw_order_type::impl_dim1) :
      blk_->stride(3, raw_order_type::impl_dim2);
  }

protected:
  typename View_block_storage<B>::type blk_;
};

template <typename B, dimension_type D>
class Redim_block<B, D, true> : public Redim_block<B, D, false>
{
  typedef Redim_block<B, D, false> base_type;
public:
  typedef typename B::ptr_type ptr_type;
  typedef typename base_type::value_type value_type;

  Redim_block(B &block) : base_type(block) {}

  void put(index_type idx, value_type val) VSIP_NOTHROW
  { redim_put(*this->blk_, idx, val, integral_constant<int, D>());}

  using base_type::ptr;
  ptr_type ptr() VSIP_NOTHROW { return this->blk_->ptr();}
};

#else

template <typename B, dimension_type D>
class Redim_block : Compile_time_assert<D == 2 || D == 3>
{
public:
  static dimension_type const dim = 1;

  typedef typename B::value_type value_type;
  typedef typename B::reference_type reference_type;
  typedef typename B::const_reference_type const_reference_type;
  typedef typename B::map_type map_type;

  typedef typename get_block_layout<B>::order_type raw_order_type;

  typedef typename B::ptr_type block_ptr_type;
  typedef typename B::const_ptr_type const_ptr_type;
  typedef typename conditional<is_const<B>::value,
			       const_ptr_type, block_ptr_type>::type ptr_type;

  Redim_block(B &block) : blk_(&block) {}
  Redim_block(Redim_block const &rb) VSIP_NOTHROW : blk_(&*rb.blk_) {}
  ~Redim_block() VSIP_NOTHROW {}

  value_type get(index_type idx) const VSIP_NOTHROW
  { return redim_get(*blk_, idx, integral_constant<int, D>());}
  void put(index_type idx, value_type val) VSIP_NOTHROW
  { redim_put(*blk_, idx, val, integral_constant<int, D>());}

  length_type size() const VSIP_NOTHROW
  { return blk_->size();}

  length_type size(dimension_type block_dim ATTRIBUTE_UNUSED, dimension_type d ATTRIBUTE_UNUSED) const VSIP_NOTHROW
  {
    assert(block_dim == 1 && d == 0);
    return blk_->size();
  }

  map_type const& map() const VSIP_NOTHROW
  { return blk_->map_;}

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}

  const_ptr_type ptr() const VSIP_NOTHROW { return blk_->ptr();}
  ptr_type ptr() VSIP_NOTHROW { return blk_->ptr();}

  stride_type stride(dimension_type block_dim ATTRIBUTE_UNUSED, dimension_type d ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  {
    assert(block_dim == 1 && d == 0);
    return D == 2 ?
      blk_->stride(2, raw_order_type::impl_dim1) :
      blk_->stride(3, raw_order_type::impl_dim2);
  }

protected:
  typename View_block_storage<B>::type blk_;
};
#endif

} // namespace vsip::impl

template <typename BlockT, dimension_type Dim>
struct get_block_layout<impl::Redim_block<BlockT, Dim> >
{
  static dimension_type const dim = 1;

  typedef row1_type                                 order_type;
  static pack_type const packing = 
    is_packing_unit_stride<get_block_layout<BlockT>::packing>::value
    ? unit_stride : any_packing;
  static storage_format_type const storage_format = get_block_layout<BlockT>::storage_format;

  typedef Layout<dim, order_type, packing, storage_format> type;
};

template <typename Block, dimension_type D>
struct supports_dda<impl::Redim_block<Block, D> >
{ static bool const value = supports_dda<Block>::value;};

namespace impl
{

// Store Redim_block by-value.
template <typename BlockT, dimension_type Dim>
struct View_block_storage<Redim_block<BlockT, Dim> >
  : By_value_block_storage<Redim_block<BlockT, Dim> >
{};

/// Reduction to redimension an expression from x-dimensional (where x > 1)
/// to 1-dimensional.
///
/// Transform expression block dimensions to 1, keeps dense blocks
/// (which are 1,x-dimensional) as is, wraps other blocks with Redim_block.
template <dimension_type NewDim>
class Redim_expr
{
public:
  template <typename BlockT>
  struct leaf_node
  {
    typedef Redim_block<BlockT, get_block_layout<BlockT>::dim> type;
  };

  template <dimension_type D, typename T, typename O, typename M>
  struct leaf_node<Dense<D, T, O, M> > { typedef Dense<D, T, O, M> type;};

  template <dimension_type D, typename T, typename O, typename M>
  struct leaf_node<Dense<D, T, O, M> const> { typedef Dense<D, T, O, M> const type;};

  template <dimension_type D, typename T, typename O> struct leaf_node<Dense<D, T, O, Local_or_global_map<D> > >
  { typedef Dense<D, T, O, Local_map> type;};

  template <dimension_type D, typename T, typename O> struct leaf_node<Dense<D, T, O, Local_or_global_map<D> > const>
  { typedef Dense<D, T, O, Local_map> const type;};

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> const>
  {
    typedef expr::Scalar<NewDim, T> const type;
  };

  template <template <typename> class O, typename B>
  struct unary_node
  {
    typedef expr::Unary<O, B, true> const type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct binary_node
  {
    typedef expr::Binary<Operation, LBlock, RBlock, true> const type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct ternary_node
  {
    typedef expr::Ternary<Operation, Block1, Block2, Block3, true> const type;
  };

  template <typename BlockT>
  struct transform
  {
    typedef typename leaf_node<BlockT>::type type;
  };

  template <template <typename> class O, typename B>
  struct transform<expr::Unary<O, B, true> const>
  {
    typedef typename unary_node<O, typename transform<B>::type>::type
    type;
  };

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  struct transform<expr::Binary<Operation, LBlock, RBlock, true> const>
  {
    typedef typename binary_node<Operation,
				 typename transform<LBlock>::type,
				 typename transform<RBlock>::type>
    ::type type;
  };

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  struct transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>
  {
    typedef typename ternary_node<Operation,
				  typename transform<Block1>::type,
				  typename transform<Block2>::type,
				  typename transform<Block3>::type>
    ::type type;
  };


  template <template <typename> class O, typename B>
  typename transform<expr::Unary<O, B, true> const>::type
  apply(expr::Unary<O, B, true> const &b)
  {
    typedef typename transform<expr::Unary<O, B, true> const>::type
      block_type;
    return block_type(b.operation(), apply(const_cast<B&>(b.arg())));
  }

  template <template <typename, typename> class Operation,
	    typename LBlock,
	    typename RBlock>
  typename transform<expr::Binary<Operation, LBlock, RBlock, true> const>::type
  apply(expr::Binary<Operation, LBlock, RBlock, true> const& blk)
  {
    typedef typename
      transform<expr::Binary<Operation, LBlock, RBlock, true> const>::type
        block_type;
    return block_type(apply(const_cast<LBlock&>(blk.arg1())),
		      apply(const_cast<RBlock&>(blk.arg2())));
  }

  template <template <typename, typename, typename> class Operation,
	    typename Block1,
	    typename Block2,
	    typename Block3>
  typename transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>::type
  apply(expr::Ternary<Operation, Block1, Block2, Block3, true> const& blk)
  {
    typedef typename
      transform<expr::Ternary<Operation, Block1, Block2, Block3, true> const>::type
        block_type;
    return block_type(apply(const_cast<Block1&>(blk.arg1())),
		      apply(const_cast<Block2&>(blk.arg2())),
		      apply(const_cast<Block3&>(blk.arg3())));
  }

  // Leaf combine function for Dense.
  template <dimension_type D, typename T, typename O, typename M>
  // typename transform<Dense<Dim0, T, OrderT, MapT> >::type&
  Dense<D, T, O, M>&
  apply(Dense<D, T, O, M> &block) const { return block;}

  // Leaf combine function for Dense.
  template <dimension_type D, typename T, typename O, typename M>
  // typename transform<Dense<Dim0, T, OrderT, MapT> >::type&
  Dense<D, T, O, M> const &
  apply(Dense<D, T, O, M> const &block) const { return block;}

  template <dimension_type D, typename T, typename O>
  // typename transform<Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> > >::type&
  Dense<D, T, O, Local_map> &
  apply(Dense<D, T, O, Local_or_global_map<D> > &block) const { return block.get_local_block();}

  template <dimension_type D, typename T, typename O>
  // typename transform<Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> > >::type&
  Dense<D, T, O, Local_map> const &
  apply(Dense<D, T, O, Local_or_global_map<D> > const &block) const { return block.get_local_block();}

  // Leaf combine function for expr::Scalar.
  template <dimension_type Dim0,
	    typename       T>
  typename transform<expr::Scalar<Dim0, T> const>::type
  apply(expr::Scalar<Dim0, T> const &block) const
  {
    return expr::Scalar<NewDim, T>(block.value());
  }


  // Leaf combine function.
  template <typename BlockT>
  typename transform<BlockT>::type
  apply(BlockT& block) const
  {
    typedef typename transform<BlockT>::type block_type;
    return block_type(block);
  }

  // Constructors.
public:
  Redim_expr() {}
};

} // namespace vsip::impl
} // namespace vsip

#endif
