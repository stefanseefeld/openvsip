/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/simd/expr_evaluator.hpp
    @author  Stefan Seefeld
    @date    2006-07-25
    @brief   VSIPL++ Library: SIMD expression evaluator proxy factory.

*/

#ifndef VSIP_IMPL_SIMD_PROXY_FACTORY_HPP
#define VSIP_IMPL_SIMD_PROXY_FACTORY_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/opt/simd/simd.hpp>
#include <vsip/opt/simd/expr_iterator.hpp>
#include <vsip/core/expr/operations.hpp>
#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/expr/binary_block.hpp>
#include <vsip/core/metaprogramming.hpp>

/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace simd
{

template <typename BlockT, bool A>
struct Proxy_factory
{
  typedef Direct_access_traits<typename BlockT::value_type> access_traits;
  typedef Proxy<access_traits, A> proxy_type;
  typedef typename adjust_layout_dim<
                     1, typename get_block_layout<BlockT>::type>::type
		layout_type;

  static bool const ct_valid = dda::Data<BlockT, dda::in>::ct_cost == 0 &&
    !is_split_block<BlockT>::value;

  static bool 
  rt_valid(BlockT const &b, int alignment)
  {
    dda::Data<BlockT, dda::in, layout_type> data(b);
    return data.stride(0) == 1 && 
      (!A ||
       Simd_traits<typename BlockT::value_type>::alignment_of(data.ptr()) ==
       alignment);
  }

  static int
  alignment(BlockT const &b)
  {
    dda::Data<BlockT, dda::in, layout_type> data(b);
    return Simd_traits<typename BlockT::value_type>::alignment_of(data.ptr());
  }

  static proxy_type
  create(BlockT const &b) 
  {
    dda::Data<BlockT, dda::in, layout_type> data(b);
    return proxy_type(data.ptr());
  }
};

template <typename T, bool A>
struct Proxy_factory<expr::Scalar<1, T> const, A>
{
  typedef Scalar_access_traits<T> access_traits;
  typedef Proxy<access_traits, A> proxy_type;
  static bool const ct_valid = true;

  static bool 
  rt_valid(expr::Scalar<1, T> const &, int) {return true;}

  static proxy_type
  create(expr::Scalar<1, T> const &b) 
  {
    return proxy_type(b.value());
  }
};

template <template <typename> class O, typename B, bool A>
struct Proxy_factory<expr::Unary<O, B, true> const, A>
{
  typedef 
    Unary_access_traits<typename Proxy_factory<B,A>::proxy_type, O>
    access_traits;
  typedef Proxy<access_traits,A> proxy_type;

  static bool const ct_valid =
    Unary_operator_map<typename B::value_type, O>::is_supported &&
    Proxy_factory<B, A>::ct_valid;

  static bool 
  rt_valid(expr::Unary<O, B, true> const &b, int alignment)
  {
    return Proxy_factory<B, A>::rt_valid(b.arg(), alignment);
  }

  static proxy_type
  create(expr::Unary<O, B, true> const &b)
  {
    return proxy_type(Proxy_factory<B, A>::create(b.arg()));
  }
};

// This proxy is specialized for unaligned blocks. If the user specifies
// ualigned(block), this is a hint to switch to an unaligned proxy.
template <typename B, bool A>
struct Proxy_factory<expr::Unary<expr::op::Unaligned, B, true> const, A>
{
  typedef typename Proxy_factory<B, false>::access_traits access_traits;
  typedef Proxy<access_traits,false> proxy_type;
  static bool const ct_valid = Proxy_factory<B,false>::ct_valid;


  static bool 
  rt_valid(expr::Unary<expr::op::Unaligned, B, true> const &b, int alignment)
  {
    return Proxy_factory<B, false>::rt_valid(b.arg(), alignment);
  }

  static proxy_type
  create(expr::Unary<expr::op::Unaligned, B, true> const &b)
  {
    return proxy_type(Proxy_factory<B, false>::create(b.arg()));
  }
};

template <template <typename, typename> class O,
	  typename LB,
	  typename RB,
	  bool A>
struct Proxy_factory<expr::Binary<O, LB, RB, true> const, A>
{
  typedef
    Binary_access_traits<typename Proxy_factory<LB, A>::proxy_type,
			 typename Proxy_factory<RB, A>::proxy_type, O> 
    access_traits;
  typedef Proxy<access_traits, A> proxy_type;
  typedef typename LB::value_type l_type;
  typedef typename RB::value_type r_type;

  static bool const ct_valid = 
    is_same<typename LB::value_type, typename RB::value_type>::value &&
    Binary_operator_map<typename LB::value_type, O>::is_supported &&
    Proxy_factory<LB, A>::ct_valid &&
    Proxy_factory<RB, A>::ct_valid;

  static bool 
  rt_valid(expr::Binary<O, LB, RB, true> const &b, int alignment)
  {
    return Proxy_factory<LB, A>::rt_valid(b.arg1(), alignment) &&
           Proxy_factory<RB, A>::rt_valid(b.arg2(), alignment);
  }

  static proxy_type
  create(expr::Binary<O, LB, RB, true> const &b)
  {
    typename Proxy_factory<LB, A>::proxy_type lp =
      Proxy_factory<LB, A>::create(b.arg1());
    typename Proxy_factory<RB, A>::proxy_type rp =
      Proxy_factory<RB, A>::create(b.arg2());

    return proxy_type(lp, rp);
  }
};

template <template <typename, typename,typename> class O,
	  typename Block1,
	  typename Block2,
	  typename Block3,
	  bool A>
struct Proxy_factory<expr::Ternary<O, Block1, Block2, Block3, true> const, A>
{
  typedef Ternary_access_traits<typename Proxy_factory<Block1, A>::proxy_type,
                                typename Proxy_factory<Block2, A>::proxy_type,
                                typename Proxy_factory<Block3, A>::proxy_type,
		 	        O> 
    access_traits;

  typedef expr::Ternary<O, Block1, Block2, Block3, true> SrcBlock;

  typedef Proxy<access_traits, A> proxy_type;
  static bool const ct_valid = 
    Ternary_operator_map<typename Block1::value_type, O>::is_supported &&
    Proxy_factory<Block1, A>::ct_valid &&
    Proxy_factory<Block2, A>::ct_valid &&
    Proxy_factory<Block3, A>::ct_valid;

  static bool 
  rt_valid(SrcBlock const &b, int alignment)
  {
    return Proxy_factory<Block1, A>::rt_valid(b.arg1(), alignment) &&
           Proxy_factory<Block2, A>::rt_valid(b.arg2(), alignment) &&
           Proxy_factory<Block3, A>::rt_valid(b.arg3(), alignment);
  }

  static proxy_type
  create(SrcBlock const &b)
  {
    typename Proxy_factory<Block1, A>::proxy_type
      b1p = Proxy_factory<Block1, A>::create(b.arg1());
    typename Proxy_factory<Block2, A>::proxy_type
      b2p = Proxy_factory<Block2, A>::create(b.arg2());
    typename Proxy_factory<Block3, A>::proxy_type
      b3p = Proxy_factory<Block3, A>::create(b.arg3());

    return proxy_type(b1p,b2p,b3p);
  }
};


} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_SIMD_PROXY_FACTORY_HPP
