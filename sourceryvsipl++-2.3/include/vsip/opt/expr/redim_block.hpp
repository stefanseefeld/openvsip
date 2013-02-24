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
#include <vsip/core/extdata.hpp>
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
  index_type idx[2];

  for (dimension_type d=2; d-->0;)
  {
    idx[d] = l_idx % blk.size(2, d);
    l_idx /= blk.size(2, d);
  }

  return blk.get(idx[0], idx[1]);
}

template <typename BlockT>
void
redim_put(BlockT &blk,
	  index_type l_idx,
	  typename BlockT::value_type value,
	  integral_constant<int, 2>)
{
  index_type idx[2];

  for (dimension_type d=2; d-->0;)
  {
    idx[d] = l_idx % blk.size(2, d);
    l_idx /= blk.size(2, d);
  }

  blk.put(idx[0], idx[1], value);
}

template <typename BlockT>
typename BlockT::value_type
redim_get(BlockT const& blk, index_type l_idx, integral_constant<int, 3>)
{
  index_type idx[3];

  for (dimension_type d=3; d-->0;)
  {
    idx[d] = l_idx % blk.size(3, d);
    l_idx /= blk.size(3, d);
  }

  return blk.get(idx[0], idx[1], idx[2]);
}

template <typename BlockT>
void
redim_put(BlockT &blk,
	  index_type l_idx,
	  typename BlockT::value_type value,
	  integral_constant<int, 3>)
{
  index_type idx[3];

  for (dimension_type d=3; d-->0;)
  {
    idx[d] = l_idx % blk.size(3, d);
    l_idx /= blk.size(3, d);
  }

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
 	  bool Mutable = Is_modifiable_block<Block>::value>
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

  typedef typename Block_layout<B>::order_type raw_order_type;

  typedef Storage<typename Block_layout<B>::complex_type, value_type>
    storage_type;
  typedef typename storage_type::type data_type;
  typedef typename storage_type::const_type const_data_type;

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

  const_data_type impl_data() const VSIP_NOTHROW
  { return blk_->impl_data();}

  stride_type impl_stride(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    assert(block_dim == 1 && d == 0);
    return D == 2 ?
      blk_->impl_stride(2, raw_order_type::impl_dim1) :
      blk_->impl_stride(3, raw_order_type::impl_dim2);
  }

protected:
  typename View_block_storage<B>::type blk_;
};

template <typename B, dimension_type D>
class Redim_block<B, D, true> : public Redim_block<B, D, false>
{
  typedef Redim_block<B, D, false> base_type;
public:
  typedef typename base_type::data_type data_type;
  typedef typename base_type::value_type value_type;

  Redim_block(B &block) : base_type(block) {}

  void put(index_type idx, value_type val) VSIP_NOTHROW
  { redim_put(*this->blk_, idx, val, integral_constant<int, D>());}

  using base_type::impl_data;
  data_type impl_data() VSIP_NOTHROW { return this->blk_->impl_data();}
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

  typedef typename Block_layout<B>::order_type raw_order_type;

  typedef Storage<typename Block_layout<B>::complex_type, value_type>
    storage_type;
  typedef typename storage_type::type data_type;
  typedef typename storage_type::const_type const_data_type;

  Redim_block(B &block) : blk_(&block) {}
  Redim_block(Redim_block const &rb) VSIP_NOTHROW : blk_(&*rb.blk_) {}
  ~Redim_block() VSIP_NOTHROW {}

  value_type get(index_type idx) const VSIP_NOTHROW
  { return redim_get(*blk_, idx, integral_constant<int, D>());}
  void put(index_type idx, value_type val) VSIP_NOTHROW
  { redim_put(*blk_, idx, val, integral_constant<int, D>());}

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

  data_type impl_data() const VSIP_NOTHROW
  { return const_cast_<data_type>(blk_->impl_data());}
  data_type impl_data() VSIP_NOTHROW { return const_cast_<data_type>(blk_->impl_data());}

  stride_type impl_stride(dimension_type block_dim, dimension_type d)
    const VSIP_NOTHROW
  {
    assert(block_dim == 1 && d == 0);
    return D == 2 ?
      blk_->impl_stride(2, raw_order_type::impl_dim1) :
      blk_->impl_stride(3, raw_order_type::impl_dim2);
  }

protected:
  typename View_block_storage<B>::type blk_;
};
#endif

template <typename       BlockT,
	  dimension_type Dim>
struct Block_layout<Redim_block<BlockT, Dim> >
{
  // Dimension: 1
  // Access   : Same
  // Order    : row1_type
  // Stride   : Stride_unit if parent Stride_unit*
  //            Stride_unknown otherwise
  // Cmplx    : Same

  static dimension_type const dim = 1;

  typedef typename Block_layout<BlockT>::access_type access_type;
  typedef row1_type                                 order_type;
  typedef typename conditional<
    Block_layout<BlockT>::pack_type::is_ct_unit_stride,
    Stride_unit, Stride_unknown>::type pack_type;
  typedef typename Block_layout<BlockT>::complex_type complex_type;

  typedef Layout<dim, order_type, pack_type, complex_type> layout_type;
};

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
    typedef Redim_block<BlockT, Block_layout<BlockT>::dim> type;
  };

  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT,
	    typename       MapT>
  struct leaf_node<Dense<Dim0, T, OrderT, MapT> >
  {
    typedef Dense<Dim0, T, OrderT, MapT> type;
  };

  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT>
  struct leaf_node<Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> > >
  {
    typedef Dense<Dim0, T, OrderT, Local_map> type;
  };

  template <dimension_type Dim0,
	    typename       T>
  struct leaf_node<expr::Scalar<Dim0, T> >
  {
    typedef expr::Scalar<NewDim, T> type;
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
  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT,
	    typename       MapT>
  // typename transform<Dense<Dim0, T, OrderT, MapT> >::type&
  Dense<Dim0, T, OrderT, MapT>&
  apply(Dense<Dim0, T, OrderT, MapT>& block) const
  {
    return block;
  }

  template <dimension_type Dim0,
	    typename       T,
	    typename       OrderT>
  // typename transform<Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> > >::type&
  Dense<Dim0, T, OrderT, Local_map>&
  apply(Dense<Dim0, T, OrderT, Local_or_global_map<Dim0> >& block) const
  {
    return block.get_local_block();
  }

  // Leaf combine function for expr::Scalar.
  template <dimension_type Dim0,
	    typename       T>
  typename transform<expr::Scalar<Dim0, T> >::type
  apply(expr::Scalar<Dim0, T> & block) const
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
