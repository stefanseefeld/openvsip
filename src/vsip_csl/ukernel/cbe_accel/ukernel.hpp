/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description: User Kernel accelerator-side base API

#ifndef VSIP_CSL_UKERNEL_CBE_ACCEL_UKERNEL_HPP
#define VSIP_CSL_UKERNEL_CBE_ACCEL_UKERNEL_HPP

#include <vsip/opt/ukernel/cbe_accel/ukernel.hpp>

namespace vsip_csl
{
namespace ukernel
{
struct null_type {};

template <class T0 = null_type, class T1 = null_type, class T2 = null_type,
	  class T3 = null_type, class T4 = null_type, class T5 = null_type,
	  class T6 = null_type, class T7 = null_type, class T8 = null_type,
	  class T9 = null_type>
class tuple;

namespace impl
{
template <typename P>
struct Kernel_base
{
  typedef P param_type;

  void init_rank(int /*rank*/, int /*nspe*/) {}
  void init(P const &) {}
  void fini() {}
};
}

/// The Kernel class template is the accelerator-side part of the User Kernel API.
///
/// Template parameters:
///
///   :P: Pre-input arguments
///   :I: Input arguments
///   :O: Output arguments
///   :Param: Parameter set for this operation
template <typename P, typename I, typename O,
	  typename Param = Empty_params>
struct Kernel;

template <typename P>
struct Kernel<tuple<>, tuple<>, tuple<>, P> : impl::Kernel_base<P>
{
  static unsigned int const pre_argc = 0;
  static unsigned int const in_argc  = 0;
  static unsigned int const out_argc = 0;
};

template <typename R, typename P>
struct Kernel<tuple<>, tuple<>, tuple<R>, P> : impl::Kernel_base<P>
{
  static unsigned int const pre_argc = 0;
  static unsigned int const in_argc  = 0;
  static unsigned int const out_argc = 1;
  typedef R out0_type;
};

template <typename A, typename R, typename P>
struct Kernel<tuple<>, tuple<A>, tuple<R>, P> : impl::Kernel_base<P>
{
  static unsigned int const pre_argc = 0;
  static unsigned int const in_argc  = 1;
  static unsigned int const out_argc = 1;
  typedef A in0_type;
  typedef R out0_type;
};

template <typename A1, typename A2, typename R, typename P>
struct Kernel<tuple<>, tuple<A1, A2>, tuple<R>, P> : impl::Kernel_base<P>
{
  static unsigned int const pre_argc = 0;
  static unsigned int const in_argc  = 2;
  static unsigned int const out_argc = 1;
  typedef A1 in0_type;
  typedef A2 in1_type;
  typedef R out0_type;
};

template <typename A1, typename A2, typename A3, typename R, typename P>
struct Kernel<tuple<>, tuple<A1, A2, A3>, tuple<R>, P> : impl::Kernel_base<P>
{
  static unsigned int const pre_argc = 0;
  static unsigned int const in_argc  = 3;
  static unsigned int const out_argc = 1;
  typedef A1 in0_type;
  typedef A2 in1_type;
  typedef A3 in2_type;
  typedef R out0_type;
};

template <typename A0, typename A1, typename R, typename P>
struct Kernel<tuple<A0>, tuple<A1>, tuple<R>, P> : impl::Kernel_base<P>
{
  static unsigned int const pre_argc = 1;
  static unsigned int const in_argc  = 1;
  static unsigned int const out_argc = 1;
  typedef A0 in0_type;
  typedef A1 in1_type;
  typedef R out0_type;
};

} // namespace vsip_csl::ukernel
} // namespace vsip_csl

#endif
