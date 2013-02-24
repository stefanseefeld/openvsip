/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/binary_block.hpp
    @author  Stefan Seefeld
    @date    2005-01-20
    @brief   VSIPL++ Library: Binary expression block class templates.
*/

#ifndef VSIP_CORE_EXPR_BINARY_BLOCK_HPP
#define VSIP_CORE_EXPR_BINARY_BLOCK_HPP

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/assign_fwd.hpp>

namespace vsip_csl
{
namespace expr
{

/// Binary expression block type.
///
/// Template parameters:
///   :Operation: Binary operation. Parametrized on value-type
///    for elementwise operation, or on block otherwise.
///   :LBlock: The first argument block.
///   :RBlock: The second argument block.
///   :Elementwise: True if this is an elementwise operation.
template <template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise = false>
class Binary;

/// Binary expression block specialization for elementwise operations.
template <template <typename, typename> class Operation,
	  typename LBlock, typename RBlock>
class Binary<Operation, LBlock, RBlock, true> : impl::Non_assignable
{
  typedef Operation<typename LBlock::value_type, typename RBlock::value_type>
  operation_type;
public:
  static dimension_type const dim = LBlock::dim;
  typedef typename operation_type::result_type value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;
  typedef typename LBlock::map_type map_type;

  Binary(LBlock const &lhs, RBlock const &rhs)
    : lhs_(lhs), rhs_(rhs) 
  {
//     assert(!Is_sized_block<LBlock>::value ||
// 	   !Is_sized_block<RBlock>::value ||
// 	   extent<dim>(lhs_) == extent<dim>(rhs_));
  }
  Binary(operation_type const &op, LBlock const &lhs, RBlock const &rhs)
    : operation_(op), lhs_(lhs), rhs_(rhs) 
  {
//     assert(!Is_sized_block<LBlock>::value ||
// 	   !Is_sized_block<RBlock>::value ||
// 	   extent<dim>(lhs_) == extent<dim>(rhs_));
  }

  length_type size() const VSIP_NOTHROW
  {
    if (!impl::Is_sized_block<LBlock>::value) return rhs_.size();
    else if (!impl::Is_sized_block<RBlock>::value) return lhs_.size();
    else return lhs_.size(); 
  }
  length_type size(dimension_type Dim, dimension_type d) const VSIP_NOTHROW
  {
    if (!impl::Is_sized_block<LBlock>::value) return rhs_.size(Dim, d); 
    else if (!impl::Is_sized_block<RBlock>::value) return lhs_.size(Dim, d); 
    else return lhs_.size(Dim, d); 
  }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const &map() const VSIP_NOTHROW { return lhs_.map();}

  operation_type const &operation() const VSIP_NOTHROW { return operation_;}
  LBlock const &arg1() const VSIP_NOTHROW {return lhs_;}
  RBlock const &arg2() const VSIP_NOTHROW {return rhs_;}

  value_type get(index_type i) const
  {
    return operation()(this->arg1().get(i),
		       this->arg2().get(i));
  }
  value_type get(index_type i, index_type j) const
  {
    return operation()(this->arg1().get(i, j),
		       this->arg2().get(i, j));
  }
  value_type get(index_type i, index_type j, index_type k) const
  {
    return operation()(this->arg1().get(i, j, k),
		       this->arg2().get(i, j, k));
  }

private:
  operation_type operation_;
  typename impl::View_block_storage<LBlock>::expr_type lhs_;
  typename impl::View_block_storage<RBlock>::expr_type rhs_;
};

/// Binary expression block specialization for non-elementwise operations.
template <template <typename, typename> class Operation,
	  typename LBlock, typename RBlock>
class Binary<Operation, LBlock, RBlock, false> : impl::Non_assignable
{
  typedef Operation<LBlock, RBlock> operation_type;
public:
  typedef Dense<operation_type::dim, typename operation_type::result_type>
  cache_type;
  typedef typename operation_type::map_type map_type;
  typedef typename operation_type::result_type value_type;
  static dimension_type const dim = operation_type::dim;

  Binary(operation_type const &op) : operation_(op) {}
  Binary(Binary const &b) : operation_(b.operation_), cache_(b.cache_) {}

  length_type size() const VSIP_NOTHROW { return operation_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return operation_.size(block_dim, d);}
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const &map() const VSIP_NOTHROW { return operation_.map();}

  operation_type const &operation() const VSIP_NOTHROW { return operation_;}
  LBlock const &arg1() const VSIP_NOTHROW {return operation_.arg1();}
  RBlock const &arg2() const VSIP_NOTHROW {return operation_.arg2();}

  value_type get(index_type i) const
  { 
    evaluate();
    return cache_->get(i);
  }
  value_type get(index_type i, index_type j) const
  { 
    evaluate();
    return cache_->get(i, j);
  }
  value_type get(index_type i, index_type j, index_type k) const
  { 
    evaluate();
    return cache_->get(i, j, k);
  }

  /// Evaluate the operation to make this type usable as an ordinary block.
  void evaluate() const
  {
    if (!cache_.get())
    {
      cache_.reset(new cache_type(impl::block_domain<dim>(operation_)));
      operation_.apply(*(cache_.get()));
    }
  }
  /// Evaluate the operation into the given result block.
  template <typename ResultBlock>
  void apply(ResultBlock &result) const
  {
    // If this expression was already evaluated,
    // just copy the cached result.
    if (cache_.get()) impl::assign<dim>(result, *cache_);
    else operation_.apply(result);
  }

  typename cache_type::const_data_type impl_data() const 
  {
    evaluate();
    return cache_->impl_data();
  }
  stride_type impl_stride(dimension_type block_dim, dimension_type d) const
  {
    evaluate();
    return cache_->impl_stride(block_dim, d);
  }

private:
  operation_type operation_;
  mutable impl::Ref_counted_ptr<cache_type> cache_;
};

} // namespace vsip_csl::expr
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

/// Specialize Is_expr_block for binary expr blocks.
template <template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct Is_expr_block<expr::Binary<Operation, LBlock, RBlock, Elementwise> >
{ static bool const value = true; };

/// Specialize View_block_storage to control how views store binary
/// expression blocks.
template <template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct View_block_storage<expr::Binary<Operation,
				 LBlock,
				 RBlock,
				 Elementwise> const>
  : By_value_block_storage<expr::Binary<Operation,
					LBlock,
					RBlock,
					Elementwise> const>
{
};

template <template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct View_block_storage<expr::Binary<Operation,
				       LBlock,
				       RBlock, Elementwise> >
{
  // No typedef provided.  A non-const expresstion template block is
  // an error.
};

/// DDA on a non-elementwise Binary works via its cache.
template <template <typename, typename> class O,
	  typename A1, typename A2>
struct Block_layout<expr::Binary<O, A1, A2> const>
  : Block_layout<typename expr::Binary<O, A1, A2>::cache_type>
{
};

template <template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct Distributed_local_block<
  expr::Binary<Operation, LBlock, RBlock, Elementwise> const>
{
  typedef expr::Binary<Operation,
		       typename Distributed_local_block<LBlock>::type,
		       typename Distributed_local_block<RBlock>::type,
		       Elementwise> const
  type;
  typedef expr::Binary<Operation,
		       typename Distributed_local_block<LBlock>::proxy_type,
		       typename Distributed_local_block<RBlock>::proxy_type,
		       Elementwise> const
  proxy_type;
};

template <template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct Distributed_local_block<
  expr::Binary<Operation, LBlock, RBlock, Elementwise> >
{
  typedef expr::Binary<Operation,
		       typename Distributed_local_block<LBlock>::type,
		       typename Distributed_local_block<RBlock>::type,
		       Elementwise>
  type;
  typedef expr::Binary<Operation,
		       typename Distributed_local_block<LBlock>::proxy_type,
		       typename Distributed_local_block<RBlock>::proxy_type,
		       Elementwise>
  proxy_type;
};

template <template <typename, typename> class Operation,
	  typename Arg1, typename Arg2>
expr::Binary<Operation,
	     typename Distributed_local_block<Arg1>::type,
	     typename Distributed_local_block<Arg2>::type,
	     true>
get_local_block(expr::Binary<Operation, Arg1, Arg2, true> const &block)
{
  typedef expr::Binary<Operation,
    typename Distributed_local_block<Arg1>::type,
    typename Distributed_local_block<Arg2>::type, true>
    block_type;

  return block_type(get_local_block(block.arg1()),
		    get_local_block(block.arg2()));
}

template <template <typename, typename> class Operation,
	  typename Arg1, typename Arg2>
expr::Binary<Operation,
	     typename Distributed_local_block<Arg1>::type,
	     typename Distributed_local_block<Arg2>::type,
	     false>
get_local_block(expr::Binary<Operation, Arg1, Arg2, false> const &block)
{
  typedef expr::Binary<Operation,
    typename Distributed_local_block<Arg1>::type,
    typename Distributed_local_block<Arg2>::type, false>
    block_type;
  typename Operation<Arg1, Arg2>::local_type 
    local_operation(block.operation().local());
  block_type local_block(local_operation);
  return local_block;
}

template <typename CombineT,
	  template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct Combine_return_type<CombineT,
			   expr::Binary<Operation, LBlock, RBlock, Elementwise> const>
{
  typedef expr::Binary<Operation,
		typename Combine_return_type<CombineT, LBlock>::tree_type,
		typename Combine_return_type<CombineT, RBlock>::tree_type,
		Elementwise> const tree_type;
  typedef tree_type type;
};

template <typename CombineT,
	  template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
struct Combine_return_type<CombineT,
			   expr::Binary<Operation, LBlock, RBlock, Elementwise> >
: Combine_return_type<CombineT,
		      expr::Binary<Operation, LBlock, RBlock, Elementwise> const>
{};

template <typename CombineT,
	  template <typename, typename> class Operation,
	  typename LBlock,
	  typename RBlock,
	  bool Elementwise>
typename Combine_return_type<CombineT,
			     expr::Binary<Operation, LBlock, RBlock, Elementwise> const>::type
apply_combine(CombineT const&                                                     combine,
	      expr::Binary<Operation, LBlock, RBlock, Elementwise> const &block)
{
  typedef typename Combine_return_type<
    CombineT, expr::Binary<Operation, LBlock, RBlock, Elementwise> const>::type
    block_type;

  return block_type(apply_combine(combine, block.arg1()),
		    apply_combine(combine, block.arg2()));
}

template <typename                            VisitorT,
	  template <typename, typename> class Operation,
	  typename                            LBlock,
	  typename                            RBlock,
	  bool Elementwise>
void
apply_leaf(VisitorT const &visitor,
	   expr::Binary<Operation, LBlock, RBlock, Elementwise> const& block)
{
  apply_leaf(visitor, block.arg1());
  apply_leaf(visitor, block.arg2());
}



template <dimension_type D,
	  typename M,
	  template <typename, typename> class O,
	  typename B1,
	  typename B2>
struct Is_par_same_map<D, M, expr::Binary<O, B1, B2, true> const>
{
  typedef expr::Binary<O, B1, B2, true> const block_type;

  static bool value(M const &map, block_type &block)
  {
    return Is_par_same_map<D, M, B1>::value(map, block.arg1()) &&
           Is_par_same_map<D, M, B2>::value(map, block.arg2());
  }
};

template <dimension_type D,
	  typename M,
	  template <typename, typename> class O,
	  typename B1,
	  typename B2>
struct Is_par_same_map<D, M, expr::Binary<O, B1, B2, false> const>
{
  typedef expr::Binary<O, B1, B2, false> const block_type;

  static bool value(M const &map, block_type &block)
  {
    return Is_par_same_map<D, M, O<B1, B2> >::value(map, block.operation());
  }
};

template <template <typename, typename> class O,
	  typename B1,
	  typename B2>
struct Is_par_reorg_ok<expr::Binary<O, B1, B2, true> const>
{
  static bool const value = Is_par_reorg_ok<B1>::value &&
                            Is_par_reorg_ok<B2>::value;
};

template <template <typename, typename> class O,
	  typename B1,
	  typename B2>
struct Is_par_reorg_ok<expr::Binary<O, B1, B2, false> const>
{
  static bool const value = false;
};

} // namespace vsip::impl
} // namespace vsip

#endif
