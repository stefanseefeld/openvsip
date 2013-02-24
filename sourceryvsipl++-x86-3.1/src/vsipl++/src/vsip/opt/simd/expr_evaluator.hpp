/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/expr_evaluator.hpp
    @author  Stefan Seefeld
    @date    2006-07-25
    @brief   VSIPL++ Library: SIMD expression evaluator logic.

*/

#ifndef VSIP_IMPL_SIMD_EXPR_EVALUATOR_HPP
#define VSIP_IMPL_SIMD_EXPR_EVALUATOR_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/simd/expr_evaluator.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/opt/simd/expr_iterator.hpp>
#include <vsip/opt/simd/proxy_factory.hpp>

namespace vsip_csl
{
namespace dispatcher
{

/// SIMD Loop Fusion evaluator for aligned expressions.
template <typename LHS, typename RHS>
struct Evaluator<op::assign<1>, be::simd_loop_fusion, void(LHS &, RHS const &)>
{
  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type
		layout_type;

  static char const* name() { return "Expr_SIMD_Loop"; }
  
  static bool const ct_valid =
    // Is SIMD supported at all ?
    impl::simd::Simd_traits<typename LHS::value_type>::is_accel &&
    // Check that direct access is possible.
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    impl::simd::Proxy_factory<RHS const, true>::ct_valid &&
    // Only allow float, double, complex<float>, and complex<double> at this time.
    (is_same<typename impl::scalar_of<typename LHS::value_type>::type, float>::value ||
     is_same<typename impl::scalar_of<typename LHS::value_type>::type, double>::value) &&
    // Make sure both sides have the same type.
    is_same<typename LHS::value_type, typename RHS::value_type>::value &&
    // Make sure the left side is not a complex split block.
    !impl::is_split_block<LHS>::value;


  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    using namespace impl;
    dda::Data<LHS, dda::out, layout_type> data(lhs);
    return (data.stride(0) == 1 &&
      simd::Proxy_factory<RHS const, true>::rt_valid(rhs, 
      simd::Simd_traits<typename LHS::value_type>::alignment_of(data.ptr())));
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    using namespace impl;
    typedef typename simd::LValue_access_traits<typename LHS::value_type> WAT;
    typedef typename simd::Proxy_factory<RHS const, true>::proxy_type proxy_type;

    length_type const vec_size =
      simd::Simd_traits<typename LHS::value_type>::vec_size;
    dda::Data<LHS, dda::out, layout_type> data(lhs);

    simd::Proxy<WAT,true> lp(data.ptr());
    proxy_type rp(simd::Proxy_factory<RHS const, true>::create(rhs));

    length_type const size = data.size(0);
    length_type n = size;

    // loop using proxy interface. This generates the best code
    // with gcc 3.4 (with gcc 4.1 the difference to the first case
    // above is negligible).

    // First, deal with unaligned pointers
    typename dda::Data<LHS, dda::out, layout_type>::ptr_type ptr = data.ptr();
    while(simd::Simd_traits<typename LHS::value_type>::alignment_of(ptr) && n > 0)
    {
      lhs.put(size-n, rhs.get(size-n));
      --n;
      ++ptr;
      lp.increment_by_element(1);
      rp.increment_by_element(1);
    }

    while (n >= vec_size)
    {
      lp.store(rp.load());
      n -= vec_size;
      lp.increment();
      rp.increment();
    }

    // Process the remainder, using simple loop fusion.
    for (index_type i = size - n; i != size; ++i)
      lhs.put(i, rhs.get(i));
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
