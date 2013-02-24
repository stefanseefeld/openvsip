/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef tests_udk_udk_hpp_
#define tests_udk_udk_hpp_

#include <vsip/core/udk/support.hpp>
#include <vsip/core/view_traits.hpp>
#include <vsip_csl/dda.hpp>
#include <memory>

namespace test
{
namespace udk = vsip_csl::udk;
namespace dda = vsip_csl::dda;

/// A channel factory.
template <typename T> struct channel_factory;

/// A channel factory for 'in' blocks.
template <typename Block>
struct channel_factory<udk::in<Block> >
{
  static vsip::dimension_type const dim = Block::dim;
  typedef typename Block::value_type value_type;
  typedef typename vsip::impl::view_of<Block>::const_type view_type;
  typedef dda::Data<Block, dda::in> dda_type;
  typedef dda_type &arg_type;
  typedef std::auto_ptr<dda_type> type;
  static type create(view_type v)
  {
    return type(new dda_type(v.block()));
  }
};

/// A channel factory for 'out' blocks.
template <typename Block>
struct channel_factory<udk::out<Block> >
{
  static vsip::dimension_type const dim = Block::dim;
  typedef typename Block::value_type value_type;
  typedef typename vsip::impl::view_of<Block>::type view_type;
  typedef dda::Data<Block, dda::out> dda_type;
  typedef dda_type &arg_type;
  typedef std::auto_ptr<dda_type> type;
  static type create(view_type v)
  {
    return type(new dda_type(v.block()));
  }
};

template <typename Args> struct make_callable;

template <typename A> 
struct make_callable<udk::tuple<A> >
{
  typedef void(*callable)(typename channel_factory<A>::arg_type);
};

template <typename A1, typename A2> 
struct make_callable<udk::tuple<A1, A2> >
{
  typedef void(*callable)(typename channel_factory<A1>::arg_type,
			  typename channel_factory<A2>::arg_type);
};

template <typename A1, typename A2, typename A3> 
struct make_callable<udk::tuple<A1, A2, A3> >
{
  typedef void(*callable)(typename channel_factory<A1>::arg_type,
			  typename channel_factory<A2>::arg_type,
			  typename channel_factory<A3>::arg_type);
};
} // namespace test

namespace vsip_csl
{
namespace udk
{
namespace impl
{

template <typename Args>
struct Policy<target::test, Args>
{
  /// The Task constructor argument type
  typedef typename test::make_callable<Args>::callable function_type;

  /// The Task::execute argument types
  template <size_t N> 
  struct arg 
  {
    typedef typename test::channel_factory<
      typename element<N, Args>::type>::view_type type;
  };
  /// The Task::execute channel types
  template <size_t N> 
  struct channel
  {
    typedef test::channel_factory<
      typename element<N, Args>::type> type;
  };
  
  static void start() {}
  static void wait() {}
};

} // namespace vsip_csl::udk::impl
} // namespace vsip_csl::udk
} // namespace vsip_csl

#endif
