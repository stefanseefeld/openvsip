/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator harness

#ifndef vsip_csl_pi_iterator_hpp_
#define vsip_csl_pi_iterator_hpp_

#include <vsip/support.hpp>
#include <vsip/core/c++0x.hpp>
#include <vsip/core/expr/operations.hpp>
#include <cassert>

namespace vsip_csl
{
namespace pi
{
using vsip::impl::integral_constant;

typedef integral_constant<int, 0> zero_type;
typedef integral_constant<int, 1> one_type;
typedef integral_constant<int, 2> two_type;
typedef integral_constant<int, 3> three_type;
typedef integral_constant<int, 4> four_type;
typedef integral_constant<int, 5> five_type;

zero_type const _0 = zero_type();
one_type const _1 = one_type();
two_type const _2 = two_type();
three_type const _3 = three_type();
four_type const _4 = four_type();
five_type const _5 = five_type();

template <int I = 0> struct Iterator {};

/// An Offset is a less efficient alternative to an iterator, 
/// encoding the relative position with a runtime parameter.
struct Offset
{
  Offset(stride_type ii) : i(ii) {}
  stride_type i;
};

template <typename T>
struct is_iterator { static bool const value = false;};

template <int I>
struct is_iterator<Iterator<I> > { static bool const value = true;};

template <>
struct is_iterator<Offset> { static bool const value = true;};

template <int I>
Iterator<I> operator+(Iterator<>, integral_constant<int, I>) 
{ return Iterator<I>();}

template <int I>
Iterator<-I> operator-(Iterator<>, integral_constant<int, I>) 
{ return Iterator<-I>();}

template <int I>
Offset operator+(Iterator<I>, stride_type o) 
{ return Offset(I + o);}

template <int I>
Offset operator-(Iterator<I>, stride_type o) 
{ return Offset(I - o);}

/// Call forward declaration.
template <typename Block,
	  typename I = void,
	  typename J = void,
	  typename K = void> 
struct Call;

template <typename T> struct is_expr { static bool const value = false;};

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
