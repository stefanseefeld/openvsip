//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_binary_hpp_
#define ovxx_expr_binary_hpp_

#include <ovxx/support.hpp>
#include <ovxx/strided.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/refcounted.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/assign_fwd.hpp>

namespace ovxx
{
namespace expr
{
template <dimension_type D, typename S> class Scalar;

namespace detail
{
template <typename LBlock, typename RBlock> 
struct binary_map
{
  typedef typename LBlock::map_type map_type;
  static map_type const &get(LBlock const &l, RBlock const &) { return l.map();}
};

template <dimension_type D, typename S, typename Block> 
struct binary_map<Scalar<D, S> const, Block>
{
  typedef typename Block::map_type map_type;
  static map_type const &get(Scalar<D, S> const &, Block const &r)
  { return r.map();}
};
} // namespace ovxx::expr::detail

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
class Binary<Operation, LBlock, RBlock, true> : ovxx::detail::nonassignable
{
  typedef Operation<typename LBlock::value_type, typename RBlock::value_type>
  operation_type;
public:
  static dimension_type const dim = LBlock::dim;
  typedef typename operation_type::result_type value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;
  typedef typename detail::binary_map<LBlock, RBlock>::map_type map_type;

  Binary(LBlock const &lhs, RBlock const &rhs)
    : lhs_(lhs), rhs_(rhs) 
  {
//     assert(!is_sized_block<LBlock>::value ||
// 	   !is_sized_block<RBlock>::value ||
// 	   extent<dim>(lhs_) == extent<dim>(rhs_));
  }
  Binary(operation_type const &op, LBlock const &lhs, RBlock const &rhs)
    : operation_(op), lhs_(lhs), rhs_(rhs) 
  {
//     assert(!is_sized_block<LBlock>::value ||
// 	   !is_sized_block<RBlock>::value ||
// 	   extent<dim>(lhs_) == extent<dim>(rhs_));
  }

  length_type size() const VSIP_NOTHROW
  {
    if (!is_sized_block<LBlock>::value) return rhs_.size();
    else if (!is_sized_block<RBlock>::value) return lhs_.size();
    else return lhs_.size(); 
  }
  length_type size(dimension_type Dim, dimension_type d) const VSIP_NOTHROW
  {
    if (!is_sized_block<LBlock>::value) return rhs_.size(Dim, d); 
    else if (!is_sized_block<RBlock>::value) return lhs_.size(Dim, d); 
    else return lhs_.size(Dim, d); 
  }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const &map() const VSIP_NOTHROW 
  { return detail::binary_map<LBlock, RBlock>::get(lhs_, rhs_);}

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
  typename block_traits<LBlock>::expr_type lhs_;
  typename block_traits<RBlock>::expr_type rhs_;
};

/// Binary expression block specialization for non-elementwise operations.
template <template <typename, typename> class Operation,
	  typename LBlock, typename RBlock>
class Binary<Operation, LBlock, RBlock, false> : ovxx::detail::nonassignable
{
  typedef Operation<LBlock, RBlock> operation_type;
public:
  typedef Strided<operation_type::dim, typename operation_type::result_type>
  cache_type;
  typedef typename operation_type::map_type map_type;
  typedef typename operation_type::result_type value_type;
  typedef value_type &reference_type;
  typedef value_type const &const_reference_type;
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
  operation_type operation_;
  mutable ovxx::refcounted_ptr<cache_type> cache_;
};

} // namespace ovxx::expr

template <template <typename, typename> class O,
	  typename L, typename R, bool E>
struct is_expr_block<expr::Binary<O, L, R, E> >
{ static bool const value = true;};

template <template <typename, typename> class O,
	  typename L, typename R, bool E>
struct block_traits<expr::Binary<O, L, R, E> const>
  : by_value_traits<expr::Binary<O, L, R, E> const>
{};

template <template <typename, typename> class O,
	  typename L, typename R, bool E>
struct block_traits<ovxx::expr::Binary<O, L, R, E> >
{};

template <template <typename, typename> class O,
	  typename L, typename R, bool E>
struct distributed_local_block<expr::Binary<O, L, R, E> const>
{
  typedef expr::Binary<O,
		       typename distributed_local_block<L>::type,
		       typename distributed_local_block<R>::type,
		       E> const
    type;
  typedef expr::Binary<O,
		       typename distributed_local_block<L>::proxy_type,
		       typename distributed_local_block<R>::proxy_type,
		       E> const
    proxy_type;
};

template <template <typename, typename> class O,
	  typename L, typename R, bool E>
struct distributed_local_block<expr::Binary<O, L, R, E> >
{
  typedef expr::Binary<O,
		       typename distributed_local_block<L>::type,
		       typename distributed_local_block<R>::type,
		       E>
    type;
  typedef expr::Binary<O,
		       typename distributed_local_block<L>::proxy_type,
		       typename distributed_local_block<R>::proxy_type,
		       E>
    proxy_type;
};

} // namespace ovxx

namespace vsip
{

/// DDA on a non-elementwise Binary works via its cache.
template <template <typename, typename> class O,
	  typename A1, typename A2>
struct get_block_layout<ovxx::expr::Binary<O, A1, A2> const>
  : get_block_layout<typename ovxx::expr::Binary<O, A1, A2>::cache_type>
{
};

} // namespace vsip
namespace ovxx 
{ 
namespace detail 
{

template <template <typename, typename> class Operation,
	  typename Arg1, typename Arg2>
expr::Binary<Operation,
	     typename distributed_local_block<Arg1>::type,
	     typename distributed_local_block<Arg2>::type,
	     true>
get_local_block(expr::Binary<Operation, Arg1, Arg2, true> const &block)
{
  typedef expr::Binary<Operation,
		       typename distributed_local_block<Arg1>::type,
		       typename distributed_local_block<Arg2>::type, true>
    block_type;

  return block_type(get_local_block(block.arg1()),
		    get_local_block(block.arg2()));
}

template <template <typename, typename> class Operation,
	  typename Arg1, typename Arg2>
expr::Binary<Operation,
	     typename distributed_local_block<Arg1>::type,
	     typename distributed_local_block<Arg2>::type,
	     false>
get_local_block(expr::Binary<Operation, Arg1, Arg2, false> const &block)
{
  typedef expr::Binary<Operation,
		       typename distributed_local_block<Arg1>::type,
		       typename distributed_local_block<Arg2>::type, false>
    block_type;
  typename Operation<Arg1, Arg2>::local_type 
    local_operation(block.operation().local());
  block_type local_block(local_operation);
  return local_block;
}

template <template <typename, typename> class O, typename B1, typename B2, bool E>
void assert_local(expr::Binary<O, B1, B2, E> const &, index_type) {}

} // namespace ovxx::detail

namespace parallel
{
template <dimension_type D,
	  typename M,
	  template <typename, typename> class O,
	  typename B1,
	  typename B2>
bool has_same_map(M const &map, expr::Binary<O, B1, B2, true> const &block)
{
  return
    has_same_map<D>(map, block.arg1()) && 
    has_same_map<D>(map, block.arg2());
};

template <dimension_type D,
	  typename M,
	  template <typename, typename> class O,
	  typename B1,
	  typename B2>
bool has_same_map(M const &map, expr::Binary<O, B1, B2, false> const &block)
{
  return has_same_map<D>(map, block.operation());
};

template <template <typename, typename> class O,
	  typename B1,
	  typename B2>
struct is_reorg_ok<expr::Binary<O, B1, B2, true> const>
{
  static bool const value = is_reorg_ok<B1>::value &&
                            is_reorg_ok<B2>::value;
};

template <template <typename, typename> class O,
	  typename B1,
	  typename B2>
struct is_reorg_ok<expr::Binary<O, B1, B2, false> const>
{
  static bool const value = false;
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
