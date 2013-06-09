//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_unary_hpp_
#define ovxx_expr_unary_hpp_

#include <ovxx/support.hpp>
#include <ovxx/strided.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/refcounted.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/assign_fwd.hpp>

namespace ovxx
{
namespace expr
{

/// Unary expression block type.
///
/// Template parameters:
///   :Operation: Unary operation. Parametrized on value-type
///               for elementwise operation, or on block otherwise.
///   :Block: The argument block.
///   :Elementwise: True if this is an elementwise operation.
template <template <typename> class Operation,
	  typename Block,
	  bool Elementwise = false>
class Unary;

/// Unary expression block specialization for elementwise operations.
template <template <typename> class Operation, typename Block>
class Unary<Operation, Block, true> : ovxx::detail::nonassignable
{
  typedef typename Block::value_type argument_type;
  typedef Operation<argument_type> operation_type;

public:
  static dimension_type const dim = Block::dim;
  typedef typename operation_type::result_type value_type;

  typedef value_type& reference_type;
  typedef value_type const& const_reference_type;
  typedef typename Block::map_type map_type;

  Unary(Block const& block) : block_(block) {}
  Unary(operation_type const &op, Block const &block)
    : operation_(op), block_(block) {}

  length_type size() const VSIP_NOTHROW { return block_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return block_.size(block_dim, d);}
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW { return block_.map();}

  operation_type const &operation() const { return operation_;}
  Block const &arg() const VSIP_NOTHROW {return block_;}

  value_type get(index_type i) const
  { return this->operation()(this->arg().get(i));}
  value_type get(index_type i, index_type j) const
  { return this->operation()(this->arg().get(i, j));}
  value_type get(index_type i, index_type j, index_type k) const
  { return this->operation()(this->arg().get(i, j, k));}

private:
  Operation<argument_type> operation_;
  typename block_traits<Block>::expr_type block_;
};


/// Unary expression block specialization for non-elementwise operations.
template <template <typename> class Operation,
	  typename Block>
class Unary<Operation, Block, false> : ovxx::detail::nonassignable
{
public:
  static dimension_type const dim = Operation<Block>::dim;
  typedef typename Operation<Block>::result_type value_type;

  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;
  typedef typename Block::map_type map_type;

  typedef Strided<Operation<Block>::dim,
		  typename Operation<Block>::result_type> cache_type;

  Unary(Operation<Block> const &op) : operation_(op) {}
  Unary(Unary const &b) : operation_(b.operation_), cache_(b.cache_) {}

  length_type size() const VSIP_NOTHROW { return operation_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const VSIP_NOTHROW
  { return operation_.size(block_dim, d);}
  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const &map() const VSIP_NOTHROW { return operation_.map();}

  Operation<Block> const &operation() const { return operation_;}
  Block const &arg() const VSIP_NOTHROW {return operation_.arg();}

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
      cache_.reset(new cache_type(block_domain<dim>(operation_)));
      operation_.apply(*(cache_.get()));
    }
  }
  /// Evaluate the operation into the given result block.
  template <typename ResultBlock>
  void apply(ResultBlock &result) const
  {
    // If this expression was already evaluated,
    // just copy the cached result.
    if (cache_.get()) assign<dim>(result, *cache_);
    else operation_.apply(result);
  }

  typename cache_type::const_ptr_type ptr() const 
  {
    evaluate();
    return cache_->ptr();
  }
  stride_type stride(dimension_type block_dim, dimension_type d) const
  {
    evaluate();
    return cache_->stride(block_dim, d);
  }

private:
  Operation<Block> operation_;
  mutable ovxx::refcounted_ptr<cache_type> cache_;
};

} // namespace ovxx::expr

template <template <typename> class O, typename B, bool E>
struct is_expr_block<expr::Unary<O, B, E> >
{ static bool const value = true; };

template <template <typename> class O, typename B, bool E>
struct block_traits<expr::Unary<O, B, E> const>
  : by_value_traits<expr::Unary<O, B, E> const>
{};

template <template <typename> class O, typename B, bool E>
struct block_traits<expr::Unary<O, B, E> > 
{};

template <template <typename> class O, typename B, bool E>
struct distributed_local_block<expr::Unary<O, B, E> const>
{
  typedef expr::Unary<O, typename distributed_local_block<B>::type, E> const
  type;
  typedef expr::Unary<O, typename distributed_local_block<B>::proxy_type, E> const
  proxy_type;
};

template <template <typename> class O, typename B, bool E>
struct distributed_local_block<expr::Unary<O, B, E> >
{
  typedef expr::Unary<O, typename distributed_local_block<B>::type, E> 
  type;
  typedef expr::Unary<O, typename distributed_local_block<B>::proxy_type, E>
  proxy_type;
};

namespace detail 
{

template <template <typename> class O, typename B>
expr::Unary<O, typename distributed_local_block<B>::type, true>
get_local_block(expr::Unary<O, B, true> const &block)
{
  typedef expr::Unary<O, typename distributed_local_block<B>::type, true>
    block_type;

  return block_type(get_local_block(block.arg()));
}

template <template <typename> class O, typename B>
expr::Unary<O, typename distributed_local_block<B>::type, false>
get_local_block(expr::Unary<O, B, false> const &block)
{
  typedef expr::Unary<O, typename distributed_local_block<B>::type, false>
    block_type;
  typename O<B>::local_type local_operation(block.operation().local());
  block_type local_block(local_operation);
  return local_block;
}

template <template <typename> class O, typename B, bool E>
void assert_local(expr::Unary<O, B, E> const &, index_type) {}

  
} // namespace ovxx::detail

namespace parallel
{
template <dimension_type D,
	  typename M,
	  template <typename> class O, typename B>
bool has_same_map(M const &map, expr::Unary<O, B, true> const &block)
{
  return has_same_map<D>(map, block.arg());
};

template <dimension_type D,
	  typename M,
	  template <typename> class O, typename B>
bool has_same_map(M const &map, expr::Unary<O, B, false> const &block)
{
  return has_same_map<D>(map, block.operation());
};

template <template <typename> class O, typename B>
struct is_reorg_ok<expr::Unary<O, B, true> const>
{
  static bool const value = is_reorg_ok<B>::value;
};

template <template <typename> class O, typename B>
struct is_reorg_ok<expr::Unary<O, B, false> const>
{
  static bool const value = false;
};

} // namespace ovxx::parallel
} // namespace ovxx

namespace vsip 
{

/// DDA on a non-elementwise Unary works via its cache.
template <template <typename> class O, typename A>
struct get_block_layout<ovxx::expr::Unary<O, A> const>
  : get_block_layout<typename ovxx::expr::Unary<O, A>::cache_type>
{
};

} // namespace vsip

#endif
