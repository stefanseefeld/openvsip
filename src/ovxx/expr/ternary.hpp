//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_ternary_hpp_
#define ovxx_expr_ternary_hpp_

#include <ovxx/support.hpp>
#include <ovxx/strided.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/refcounted.hpp>
#include <ovxx/detail/noncopyable.hpp>
#include <ovxx/assign_fwd.hpp>
#include <memory>

namespace ovxx
{
namespace expr
{
template <dimension_type D, typename S> class Scalar;

namespace detail
{
template <typename Block1, typename Block2, typename Block3> 
struct Ternary_map
{
  typedef typename Block1::map_type map_type;
  static map_type const &get(Block1 const &b1, Block2 const &, Block3 const &)
  { return b1.map();}
};

template <dimension_type D, typename S, typename Block2, typename Block3> 
struct Ternary_map<expr::Scalar<D, S> const, Block2, Block3>
{
  typedef typename Block2::map_type map_type;
  static map_type const &get(expr::Scalar<D, S> const &, Block2 const &b2, Block3 const &)
  { return b2.map();}
};
} // namespace ovxx::expr::detail

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
class Ternary<Operation, Block1, Block2, Block3, true> : ovxx::detail::nonassignable
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
  typedef typename detail::Ternary_map<Block1, Block2, Block3>::map_type map_type;

  Ternary(Block1 const &first, Block2 const &second, Block3 const &third)
  : first_(first), second_(second), third_(third)
  {
    OVXX_PRECONDITION(!is_sized_block<Block1>::value ||
		      !is_sized_block<Block2>::value ||
		      !is_sized_block<Block3>::value ||
		      (extent<dim>(first_) == extent<dim>(second_) &&
		       extent<dim>(first_) == extent<dim>(third_)));
  }
  Ternary(operation_type const &op,
	  Block1 const &first, Block2 const &second, Block3 const &third)
  : operation_(op), first_(first), second_(second), third_(third)
  {
    OVXX_PRECONDITION(!is_sized_block<Block1>::value ||
		      !is_sized_block<Block2>::value ||
		      !is_sized_block<Block3>::value ||
		      (extent<dim>(first_) == extent<dim>(second_) &&
		       extent<dim>(first_) == extent<dim>(third_)));
  }

  length_type size() const VSIP_NOTHROW
  {
    if (is_sized_block<Block1>::value) return first_.size();
    else if (is_sized_block<Block2>::value) return second_.size();
    else return third_.size();
  }
  length_type size(dimension_type Dim, dimension_type d) const VSIP_NOTHROW
  {
    if (is_sized_block<Block1>::value) return first_.size(Dim, d);
    else if (is_sized_block<Block2>::value) return second_.size(Dim, d);
    else return third_.size(Dim, d);
  }

  void increment_count() const VSIP_NOTHROW {}
  void decrement_count() const VSIP_NOTHROW {}
  map_type const& map() const VSIP_NOTHROW 
  { return detail::Ternary_map<Block1, Block2, Block3>::get(first_, second_, third_);}

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
  typename block_traits<Block1>::expr_type first_;
  typename block_traits<Block2>::expr_type second_;
  typename block_traits<Block3>::expr_type third_;
};

/// Ternary expression block specialization for non-elementwise operations.
template <template <typename, typename, typename> class Operation,
	  typename Block1, typename Block2, typename Block3>
class Ternary<Operation, Block1, Block2, Block3, false> : ovxx::detail::nonassignable
{
  typedef Operation<Block1, Block2, Block3> operation_type;
public:
  typedef Strided<operation_type::dim, typename operation_type::result_type>
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

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct is_expr_block<expr::Ternary<O, B1, B2, B3, E> >
{ static bool const value = true;};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct block_traits<ovxx::expr::Ternary<O, B1, B2, B3, E> const>
: by_value_traits<ovxx::expr::Ternary<O, B1, B2, B3, E> const>
{};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct block_traits<ovxx::expr::Ternary<O, B1, B2, B3, E> >
{};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct distributed_local_block<expr::Ternary<O, B1, B2, B3, E> const>
{
  typedef expr::Ternary<O,
			typename distributed_local_block<B1>::type,
			typename distributed_local_block<B2>::type,
			typename distributed_local_block<B3>::type,
			E> const
  type;
  typedef expr::Ternary<O,
			typename distributed_local_block<B1>::proxy_type,
			typename distributed_local_block<B2>::proxy_type,
			typename distributed_local_block<B3>::proxy_type,
			E>
  const proxy_type;
};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct distributed_local_block<expr::Ternary<O, B1, B2, B3, E> >
{
  typedef expr::Ternary<O,
			typename distributed_local_block<B1>::type,
			typename distributed_local_block<B2>::type,
			typename distributed_local_block<B3>::type,
			E>
  type;
};

namespace expr
{

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
Ternary<O,
	typename distributed_local_block<B1>::type,
	typename distributed_local_block<B2>::type,
	typename distributed_local_block<B3>::type,
	true>
get_local_block(Ternary<O, B1, B2, B3, true> const &block)
{
  typedef Ternary<O,
		  typename distributed_local_block<B1>::type,
		  typename distributed_local_block<B2>::type,
		  typename distributed_local_block<B3>::type,
		  true>
    block_type;

  return block_type(get_local_block(block.arg1()),
		    get_local_block(block.arg2()),
		    get_local_block(block.arg3()));
}

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
Ternary<O,
	typename distributed_local_block<B1>::type,
	typename distributed_local_block<B2>::type,
	typename distributed_local_block<B3>::type,
	false>
get_local_block(Ternary<O, B1, B2, B3, false> const &block)
{
  typedef Ternary<O,
		  typename distributed_local_block<B1>::type,
		  typename distributed_local_block<B2>::type,
		  typename distributed_local_block<B3>::type,
		  false>
  block_type;

  typename O<B1, B2, B3>::local_type 
    local_operation(block.operation().local());
  block_type local_block(local_operation);
  return local_block;
}

template <template <typename, typename, typename> class O,
          typename B1, typename B2, typename B3, bool E>
void assert_local(Ternary<O, B1, B2, B3, E> const &, index_type) {}
  
} // namespace ovxx::expr

namespace parallel
{
template <dimension_type D,
	  typename M,
	  template <typename, typename, typename> class O,
	  typename B1,
	  typename B2,
	  typename B3>
bool has_same_map(M const &map, expr::Ternary<O, B1, B2, B3, true> const &block)
{
  return
    has_same_map<D>(map, block.arg1()) &&
    has_same_map<D>(map, block.arg2()) &&
    has_same_map<D>(map, block.arg3());
};

template <dimension_type D,
	  typename M,
	  template <typename, typename, typename> class O,
	  typename B1,
	  typename B2,
	  typename B3>
bool has_same_map(M const &map, expr::Ternary<O, B1, B2, B3, false> const &block)
{
  return has_same_map<D>(map, block.operation());
};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
struct is_reorg_ok<expr::Ternary<O, B1, B2, B3, true> const>
{
  static bool const value = is_reorg_ok<B1>::value &&
                            is_reorg_ok<B2>::value &&
                            is_reorg_ok<B3>::value;
};

template <template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3>
struct is_reorg_ok<expr::Ternary<O, B1, B2, B3, false> const>
{
  static bool const value = false;
};

} // namespace ovxx::parallel
} // namespace ovxx

namespace vsip 
{ 

/// DDA on a non-elementwise Ternary works via its cache.
template <template <typename, typename, typename> class O,
	  typename A1, typename A2, typename A3>
struct get_block_layout<ovxx::expr::Ternary<O, A1, A2, A3> const>
  : get_block_layout<typename ovxx::expr::Ternary<O, A1, A2, A3>::cache_type> {};

} // namespace vsip

#endif
