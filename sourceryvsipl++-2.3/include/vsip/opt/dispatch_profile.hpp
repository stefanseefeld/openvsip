/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/dispatch_profile.hpp
    @author  Stefan Seefeld
    @date    2008-12-03
    @brief   VSIPL++ Library: Dispatch profiler harness.
*/

#ifndef VSIP_OPT_DISPATCH_PROFILE_HPP
#define VSIP_OPT_DISPATCH_PROFILE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/profile.hpp>
#include <vsip/core/dispatch.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{

/// meta-function to map operations to features.
template <typename Operation>
struct Profile_feature
{
  static unsigned int const value = impl::profile::none;
};

/// No-profile policy
template <typename O, typename S, typename B> struct Profile_nop_policy;

template <typename O,
          typename R, typename A,
          typename B>
struct Profile_nop_policy<O, R(A), B>
{
  Profile_nop_policy(A) {}
};

template <typename O,
          typename R, typename A1, typename A2,
          typename B>
struct Profile_nop_policy<O, R(A1, A2), B>
{
  Profile_nop_policy(A1, A2) {}
};

template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename B>
struct Profile_nop_policy<O, R(A1, A2, A3), B>
{
  Profile_nop_policy(A1, A2, A3) {}
};

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename B>
struct Profile_nop_policy<O, R(A1, A2, A3, A4), B>
{
  Profile_nop_policy(A1, A2, A3, A4) {}
};

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4, typename A5,
          typename B>
struct Profile_nop_policy<O, R(A1, A2, A3, A4, A5), B>
{
  Profile_nop_policy(A1, A2, A3, A4, A5) {}
};

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6,
          typename B>
struct Profile_nop_policy<O, R(A1, A2, A3, A4, A5, A6), B>
{
  Profile_nop_policy(A1, A2, A3, A4, A5, A6) {}
};

// Profile policy
template <typename O, typename S, typename B> struct Profile_policy;

template <typename O,
          typename R, typename A,
          typename B>
struct Profile_policy<O, R(A), B>
{
  typedef impl::profile::Scope<Profile_feature<O>::value> scope_type;
  typedef Evaluator<O, B, R(A)> evaluator_type;
  Profile_policy(A a)
    : scope(evaluator_type::name(), evaluator_type::op_count(a)) {}
    
  scope_type scope;  
};

template <typename O,
          typename R, typename A1, typename A2,
          typename B>
struct Profile_policy<O, R(A1, A2), B>
{
  typedef impl::profile::Scope<Profile_feature<O>::value> scope_type;
  typedef Evaluator<O, B, R(A1, A2)> evaluator_type;
  Profile_policy(A1 a1, A2 a2)
    : scope(evaluator_type::name(), evaluator_type::op_count(a1, a2)) {}
    
  scope_type scope;  
};

template <typename O,
          typename R, typename A1, typename A2, typename A3,
          typename B>
struct Profile_policy<O, R(A1, A2, A3), B>
{
  typedef impl::profile::Scope<Profile_feature<O>::value> scope_type;
  typedef Evaluator<O, B, R(A1, A2, A3)> evaluator_type;
  Profile_policy(A1 a1, A2 a2, A3 a3)
    : scope(evaluator_type::name(), evaluator_type::op_count(a1, a2, a3)) {}
    
  scope_type scope;  
};

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4,
          typename B>
struct Profile_policy<O, R(A1, A2, A3, A4), B>
{
  typedef impl::profile::Scope<Profile_feature<O>::value> scope_type;
  typedef Evaluator<O, B, R(A1, A2, A3, A4)> evaluator_type;
  Profile_policy(A1 a1, A2 a2, A3 a3, A4 a4)
    : scope(evaluator_type::name(), evaluator_type::op_count(a1, a2, a3, a4)) {}
    
  scope_type scope;  
};

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4, typename A5,
          typename B>
struct Profile_policy<O, R(A1, A2, A3, A4, A5), B>
{
  typedef impl::profile::Scope<Profile_feature<O>::value> scope_type;
  typedef Evaluator<O, B, R(A1, A2, A3, A4, A5)> evaluator_type;
  Profile_policy(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
    : scope(evaluator_type::name(), evaluator_type::op_count(a1, a2, a3, a4, a5)) {}
    
  scope_type scope;  
};

template <typename O,
          typename R, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6,
          typename B>
struct Profile_policy<O, R(A1, A2, A3, A4, A5, A6), B>
{
  typedef impl::profile::Scope<Profile_feature<O>::value> scope_type;
  typedef Evaluator<O, B, R(A1, A2, A3, A4, A5, A6)> evaluator_type;
  Profile_policy(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
    : scope(evaluator_type::name(), evaluator_type::op_count(a1, a2, a3, a4, a5, a6)) {}
    
  scope_type scope;  
};

template <typename O,  // operation
          typename S,  // signature
          typename B>  // backend
struct Profile_policy_selector
{
  typedef typename impl::ITE_Type<
    Profile_feature<O>::value & impl::profile::mask,
    impl::As_type<Profile_policy<O, S, B> >,
    impl::As_type<Profile_nop_policy<O, S, B> > >::type type;
};


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
