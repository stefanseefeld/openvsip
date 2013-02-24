/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/expr/ternary_block.hpp
    @author  Stefan Seefeld
    @date    2005-04-25
    @brief   VSIPL++ Library: Ternary expression block class templates.
*/

#ifndef VSIP_CORE_EXPR_TERNARY_BLOCK_HPP
#define VSIP_CORE_EXPR_TERNARY_BLOCK_HPP

#include <vsip/support.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/noncopyable.hpp>
#include <vsip/core/assign_fwd.hpp>
#include <memory>

namespace vsip_csl
{
namespace expr
{

/// Ternary expression block type.
///
/// Template parameters:
///   :Operation: Ternary operation. Parametrized on value-type 
///               for elementwise operation, or on block otherwise.
///   :Block1: The first argument block.
///   :Block2: The second argument block.
///   :Block3: The third argument block.
///   :Elementwise: True if this is an elementwise operation.
template <template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3, bool Elementwise = false>
class Ternary;

/// Ternary expression block specialization for elementwise operations.
template <template <typename, typename, typename> class Operation,
	  typename Block1, typename Block2, typename Block3>
class Ternary<Operation, Block1, Block2, Block3, true> : impl::Non_assignable
{
  typedef Operation<typename Block1::value_type,
		    typename Block2::value_type,
		    typename Block3::value_type>
  operation_type;
public:
  static dimension_type const dim = Block1::dim;
  typedef typename operation_type::result_type value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;
  typedef typename Block1::map_type map_type;

  Ternary(Block1 const &first, Block2 const &second, Block3 const &third)
  : first_(first), second_(second), third_(third)
  {
    assert(!impl::Is_sized_block<Block1>::value ||
	   !impl::Is_sized_block<Block2>::value ||
	   !impl::Is_sized_block<Block3>::value ||
	   (impl::extent<dim>(first_) == impl::extent<dim>(second_) &&
	    impl::extent<dim>(first_) == impl::extent<dim>(third_)));
  }
  Ternary(operation_type const &op,
	  Block1 const &first, Block2 const &second, Block3 const &third)
  : operation_(op), first_(first), second_(second), third_(third)
  {
    assert(!impl::Is_sized_block<Block1>::value ||
	   !impl::Is_sized_block<Block2>::value ||
	   !impl::Is_sized_block<Block3>::value ||
	   (impl::extent<dim>(first_) == impl::extent<dim>(second_) &&
	    impl::extent<dim>(first_) == impl::extent<dim>(third_)));
  }

  length_type size() const VSIP_NOTHROW
  {
    if (impl::Is_sized_block<Block1>::value) return first_.size();
    else if (impl::Is_sized_block<Block2>::value) return second_.size();
    else return third_.size();
  }
  length_type size(dimension_type Dim, dimension_type d) const VSIP_NOTHROW
  {
    if (impl::Is_sized_block<Block1>::value) return first_.size(Dim, d);
    else if (impl::Is_sized_block<Block2>::value) return second_.size(Dim, d);
    else return third_.size(Dim, d);
  }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW { return first_.map();}

  operation_type const &operation() const VSIP_NOTHROW { return operation_;}
  Block1 const &arg1() const VSIP_NOTHROW {return first_;}
  Block2 const &arg2() const VSIP_NOTHROW {return second_;}
  Block3 const &arg3() const VSIP_NOTHROW {return third_;}

  value_type get(index_type i) const
  {
    return operation()(this->arg1().get(i),
		       this->arg2().get(i),
		       this->arg3().get(i));
  }
  value_type get(index_type i, index_type j) const
  {
    return operation()(this->arg1().get(i, j),
		       this->arg2().get(i, j),
		       this->arg3().get(i, j));
  }
  value_type get(index_type i, index_type j, index_type k) const
  {
    return operation()(this->arg1().get(i, j, k),
		       this->arg2().get(i, j, k),
		       this->arg3().get(i, j, k));
  }

private:
  operation_type operation_;
  typename impl::View_block_storage<Block1>::expr_type first_;
  typename impl::View_block_storage<Block2>::expr_type second_;
  typename impl::View_block_storage<Block3>::expr_type third_;
};

/// Ternary expression block specialization for non-elementwise operations.
template <template <typename, typename, typename> class Operation,
	  typename Block1, typename Block2, typename Block3>
class Ternary<Operation, Block1, Block2, Block3, false> : impl::Non_assignable
{
  typedef Operation<Block1, Block2, Block3> operation_type;
public:
  typedef Dense<operation_type::dim, typename operation_type::result_type>
  cache_type;
  typedef typename operation_type::map_type map_type;
  typedef typename operation_type::result_type value_type;
  static dimension_type const dim = operation_type::dim;

  Ternary(operation_type const &op) : operation_(op) {}
  Ternary(Ternary const &b) : operation_(b.operation_), cache_(b.cache_) {}

  length_type size() const VSIP_NOTHROW { return operation_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return operation_.size(block_dim, d);}
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const &map() const VSIP_NOTHROW { return operation_.map();}

  operation_type const &operation() const VSIP_NOTHROW { return operation_;}
  Block1 const &arg1() const VSIP_NOTHROW {return operation_.arg1();}
  Block2 const &arg2() const VSIP_NOTHROW {return operation_.arg2();}
  Block3 const &arg3() const VSIP_NOTHROW {return operation_.arg3();}

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

/// Specialize Is_expr_block for ternary expr blocks.
template <template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
struct Is_expr_block<expr::Ternary<Operation,
				   Block1,
				   Block2,
				   Block3, Elementwise> >
{ static bool const value = true; };



/// Specialize View_block_storage to control how views store binary
/// expression template blocks.
template <template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3, 
	  bool Elementwise>
struct View_block_storage<expr::Ternary<Operation,
					Block1,
					Block2,
					Block3,
					Elementwise> const>
  : By_value_block_storage<expr::Ternary<Operation,
					 Block1,
					 Block2,
					 Block3, Elementwise> const>
{};

template <template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
struct View_block_storage<expr::Ternary<Operation,
					Block1,
					Block2,
					Block3,
					Elementwise> >
{
  // No typedef provided.  A non-const expresstion template block is
  // an error.
};

/// DDA on a non-elementwise Ternary works via its cache.
template <template <typename, typename, typename> class O,
	  typename A1, typename A2, typename A3>
struct Block_layout<expr::Ternary<O, A1, A2, A3> const>
  : Block_layout<typename expr::Ternary<O, A1, A2, A3>::cache_type>
{
};

/***********************************************************************
  Parallel traits and functions
***********************************************************************/

template <template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
struct Distributed_local_block<
  expr::Ternary<Operation,
		Block1,
		Block2,
		Block3,
		Elementwise> const>
{
  typedef expr::Ternary<Operation,
			typename Distributed_local_block<Block1>::type,
			typename Distributed_local_block<Block2>::type,
			typename Distributed_local_block<Block3>::type,
			Elementwise> const
  type;
  typedef expr::Ternary<Operation,
			typename Distributed_local_block<Block1>::proxy_type,
			typename Distributed_local_block<Block2>::proxy_type,
			typename Distributed_local_block<Block3>::proxy_type,
			Elementwise>
  const proxy_type;
};

template <template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
struct Distributed_local_block<
  expr::Ternary<Operation,
		Block1,
		Block2,
		Block3, 
		Elementwise> >
{
  typedef expr::Ternary<Operation,
			typename Distributed_local_block<Block1>::type,
			typename Distributed_local_block<Block2>::type,
			typename Distributed_local_block<Block3>::type,
			Elementwise>
  type;
};



template <template <typename, typename, typename> class Operation,
	  typename Block1, typename Block2, typename Block3>
expr::Ternary<Operation,
	      typename Distributed_local_block<Block1>::type,
	      typename Distributed_local_block<Block2>::type,
	      typename Distributed_local_block<Block3>::type,
	      true>
get_local_block(expr::Ternary<Operation, Block1, Block2, Block3, true> const &block)
{
  typedef expr::Ternary<Operation,
    typename Distributed_local_block<Block1>::type,
    typename Distributed_local_block<Block2>::type,
    typename Distributed_local_block<Block3>::type,
    true>
  block_type;

  return block_type(get_local_block(block.arg1()),
		    get_local_block(block.arg2()),
		    get_local_block(block.arg3()));
}

template <template <typename, typename, typename> class Operation,
	  typename Block1, typename Block2, typename Block3>
expr::Ternary<Operation,
	      typename Distributed_local_block<Block1>::type,
	      typename Distributed_local_block<Block2>::type,
	      typename Distributed_local_block<Block3>::type,
	      false>
get_local_block(expr::Ternary<Operation, Block1, Block2, Block3, false> const &block)
{
  typedef expr::Ternary<Operation,
    typename Distributed_local_block<Block1>::type,
    typename Distributed_local_block<Block2>::type,
    typename Distributed_local_block<Block3>::type,
    false>
  block_type;

  typename Operation<Block1, Block2, Block3>::local_type 
    local_operation(block.operation().local());
  block_type local_block(local_operation);
  return local_block;
}



template <typename                  CombineT,
	  template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
struct Combine_return_type<CombineT,
			   expr::Ternary<Operation,
					 Block1,
					 Block2,
					 Block3,
					 Elementwise> const>
{
  typedef expr::Ternary<Operation,
			typename Combine_return_type<CombineT, Block1>::tree_type,
			typename Combine_return_type<CombineT, Block2>::tree_type,
			typename Combine_return_type<CombineT, Block3>::tree_type,
			Elementwise>
  const tree_type;
  typedef tree_type type;
};



template <typename                  CombineT,
	  template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
struct Combine_return_type<CombineT,
			   expr::Ternary<Operation,
					 Block1,
					 Block2,
					 Block3,
					 Elementwise> >
  : Combine_return_type<CombineT,
			expr::Ternary<Operation,
				      Block1,
				      Block2,
				      Block3,
				      Elementwise> const>
{};



template <typename CombineT,
	  template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
typename Combine_return_type<CombineT,
			     expr::Ternary<Operation,
					   Block1,
					   Block2,
					   Block3,
					   Elementwise> const>::type
apply_combine(CombineT const &combine,
	      expr::Ternary<Operation, Block1, Block2, Block3, Elementwise> const &block)
{
  typedef typename Combine_return_type<CombineT,
    expr::Ternary<Operation, Block1, Block2, Block3, Elementwise> const>::type
    block_type;

  return block_type(apply_combine(combine, block.arg1()),
		    apply_combine(combine, block.arg2()),
		    apply_combine(combine, block.arg3()));
}



template <typename VisitorT,
	  template <typename, typename, typename> class Operation,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool Elementwise>
void
apply_leaf(VisitorT const &visitor,
	   expr::Ternary<Operation, Block1, Block2, Block3, Elementwise> const &block)
{
  apply_leaf(visitor, block.arg1());
  apply_leaf(visitor, block.arg2());
  apply_leaf(visitor, block.arg3());
}



template <dimension_type D,
	  typename M,
	  template <typename, typename, typename> class O,
	  typename B1,
	  typename B2,
	  typename B3>
struct Is_par_same_map<D, M, expr::Ternary<O, B1, B2, B3, true> const>
{
  typedef expr::Ternary<O, B1, B2, B3, true> const block_type;

  static bool value(M const &map, block_type &block)
  {
    return Is_par_same_map<D, M, B1>::value(map, block.arg1()) &&
           Is_par_same_map<D, M, B2>::value(map, block.arg2()) &&
           Is_par_same_map<D, M, B3>::value(map, block.arg3());
  }
};

template <dimension_type D,
	  typename M,
	  template <typename, typename, typename> class O,
	  typename B1,
	  typename B2,
	  typename B3>
struct Is_par_same_map<D, M, expr::Ternary<O, B1, B2, B3, false> const>
{
  typedef expr::Ternary<O, B1, B2, B3, false> const block_type;

  static bool value(M const &map, block_type &block)
  {
    return Is_par_same_map<D, M, O<B1, B2, B3> >::value(map, block.operation());
  }
};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
struct Is_par_reorg_ok<expr::Ternary<O, B1, B2, B3, true> const>
{
  static bool const value = Is_par_reorg_ok<B1>::value &&
                            Is_par_reorg_ok<B2>::value &&
                            Is_par_reorg_ok<B3>::value;
};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
struct Is_par_reorg_ok<expr::Ternary<O, B1, B2, B3, false> const>
{
  static bool const value = false;
};

} // namespace vsip::impl
} // namespace vsip

#endif
