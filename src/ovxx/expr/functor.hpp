//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_functor_hpp_
#define ovxx_expr_functor_hpp_

#include <ovxx/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/expr/unary.hpp>

namespace ovxx
{
namespace expr
{

// Convenience base class for non-elementwise unary functors.
template <typename B>
class Unary_functor
{
public:
  static dimension_type const dim = B::dim;
  typedef typename B::value_type value_type;
  typedef value_type result_type;
  typedef typename B::map_type map_type;

  Unary_functor(B const &a) : arg_(a) {}

  B const &arg() const { return arg_;}
  length_type size() const { return arg_.size();}
  length_type size(dimension_type block_dim, dimension_type d) const
  { return arg_.size(block_dim, d);}
  map_type const &map() const { return arg_.map();}

  // Apply this functor and pass the result back in 'r'.
  // This needs to be provided by the implementor.
  template <typename R>
  void apply(R &r) const {}

private:
  typename block_traits<B>::expr_type arg_;
};

template <typename B1, typename B2>
class Binary_functor
{
public:
  // This implementation is very simple-minded,
  // and therefore only works for a restricted set
  // of cases:
  // It assumes both blocks have the same dimensionality
  // and map.
  static dimension_type const dim = B1::dim;
  typedef typename B1::map_type map_type;
    
  Binary_functor(B1 const &a1, B2 const &a2)
    : arg1_(a1), arg2_(a2) 
  {
  }

  B1 const &arg1() const { return arg1_;}
  B2 const &arg2() const { return arg2_;}
  length_type size() const { return arg1.size();}
  length_type size(dimension_type block_dim, dimension_type d) const
  { return arg1.size(block_dim, d);}
  map_type const &map() const { return arg1.map();}

  // Apply this functor and pass the result back in 'r'.
  // This needs to be provided by the implementor.
  template <typename R>
  void apply(R &r) const {}

private:
  typename block_traits<B1>::expr_type arg1_;
  typename block_traits<B2>::expr_type arg2_;
};

} // namespace ovxx::expr
} // namespace ovxx

#endif
