/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator reduction support

#ifndef vsip_csl_pi_reductions_hpp_
#define vsip_csl_pi_reductions_hpp_

#include <vsip_csl/pi/expr.hpp>

namespace vsip_csl
{
namespace pi
{

template <typename B> 
struct Alltrue
{
  typedef typename B::value_type result_type;
  
  result_type operator()(B const &block) const
  {
    typedef typename B::value_type value_type;
    typedef typename vsip::impl::view_of<B>::type view_type;
    view_type view(const_cast<B&>(block));
    return alltrue(view);
  }
};

template <typename B> 
struct Anytrue
{
  typedef typename B::value_type result_type;
  
  result_type operator()(B const &block) const
  {
    typedef typename B::value_type value_type;
    typedef typename vsip::impl::view_of<B>::type view_type;
    view_type view(const_cast<B&>(block));
    return anytrue(view);
  }
};

template <typename B> 
struct Sumval
{
  typedef typename B::value_type result_type;
  
  result_type operator()(B const &block) const
  {
    typedef typename B::value_type value_type;
    typedef typename vsip::impl::view_of<B>::type view_type;
    view_type view(const_cast<B&>(block));
    return sumval(view);
  }
};

template <typename B> 
struct Meanval
{
  typedef typename B::value_type result_type;
  
  result_type operator()(B const &block) const
  {
    typedef typename B::value_type value_type;
    typedef typename vsip::impl::view_of<B>::type view_type;
    view_type view(const_cast<B&>(block));
    return meanval(view);
  }
};

template <typename B> 
struct Sumsqval
{
  typedef typename B::value_type result_type;
  
  result_type operator()(B const &block) const
  {
    typedef typename B::value_type value_type;
    typedef typename vsip::impl::view_of<B>::type view_type;
    view_type view(const_cast<B&>(block));
    return sumsqval(view);
  }
};

template <typename B> 
struct Meansqval
{
  typedef typename B::value_type result_type;
  
  result_type operator()(B const &block) const
  {
    typedef typename B::value_type value_type;
    typedef typename vsip::impl::view_of<B>::type view_type;
    view_type view(const_cast<B&>(block));
    return meansqval(view);
  }
};

/// PI reduction overloads. C is a call-expression, such as
/// `pi::Call<B, I, whole_domain_type>` (`matrix.row(i)`) or
/// `pi::Call<B, whole_domain_type, J>` (`matrix.col(j)`).

template <typename C>
typename enable_if<pi::is_call<C>, pi::Unary<Alltrue, C> >::type
alltrue(C const &call) { return pi::Unary<Alltrue, C>(call);}

template <typename C>
typename enable_if<pi::is_call<C>, pi::Unary<Anytrue, C> >::type
anytrue(C const &call) { return pi::Unary<Anytrue, C>(call);}

template <typename C>
typename enable_if<pi::is_call<C>, pi::Unary<Meanval, C> >::type
meanval(C const &call) { return pi::Unary<Meanval, C>(call);}

template <typename C>
typename enable_if<pi::is_call<C>, pi::Unary<Sumval, C> >::type
sumval(C const &call) { return pi::Unary<Sumval, C>(call);}

template <typename C>
typename enable_if<pi::is_call<C>, pi::Unary<Meansqval, C> >::type
meansqval(C const &call) { return pi::Unary<Meansqval, C>(call);}

template <typename C>
typename enable_if<pi::is_call<C>, pi::Unary<Sumsqval, C> >::type
sumsqval(C const &call) { return pi::Unary<Sumsqval, C>(call);}

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
