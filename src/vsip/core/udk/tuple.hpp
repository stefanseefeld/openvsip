/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef vsip_core_udk_tuple_hpp_
#define vsip_core_udk_tuple_hpp_

#include <vsip/core/c++0x.hpp>

namespace vsip_csl
{
namespace udk
{
struct null_type {};

namespace impl
{
template <typename Head, typename Tail> 
struct cons
{
  typedef Head head_type;
  typedef Tail tail_type;
};

template <typename H>
struct cons<H, null_type>
{

  typedef H head_type;
  typedef null_type tail_type;
};

template <typename T0 = null_type,
	  typename T1 = null_type,
	  typename T2 = null_type,
	  typename T3 = null_type,
	  typename T4 = null_type,
          typename T5 = null_type,
	  typename T6 = null_type,
	  typename T7 = null_type,
	  typename T8 = null_type,
	  typename T9 = null_type>
struct make_cons
{
  typedef cons<T0,
               typename make_cons<T1, T2, T3, T4, T5,
				  T6, T7, T8, T9>::type
              > type;
};

template <>
struct make_cons<>
{
  typedef null_type type;
};

} // vsip_csl::udk::impl

/// This tuple template should emulate the C++0x std::tuple template as
/// closely as possible, as far as our use of it is concerned (i.e., as a
/// compile-time entity.
template <typename T0 = null_type,
	  typename T1 = null_type,
	  typename T2 = null_type,
	  typename T3 = null_type,
	  typename T4 = null_type,
          typename T5 = null_type,
	  typename T6 = null_type,
	  typename T7 = null_type,
	  typename T8 = null_type,
	  typename T9 = null_type>
struct tuple : impl::make_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type {};

/// Access the Nth element of a tuple.
template<int N, typename T>
struct element
{
private:
  typedef typename T::tail_type next_type;
public:
  typedef typename element<N-1, next_type>::type type;
};

template<typename T>
struct element<0,T>
{
  typedef typename T::head_type type;
};

} // namespace vsip_csl::udk
} // namespace vsip_csl

#endif
