/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/ukernel/host/ukernel.hpp
    @author  Jules Bergmann
    @date    2008-08-20
    @brief   VSIPL++ Library: User-defined Kernels.
*/

#ifndef VSIP_CSL_UKERNEL_HOST_UKERNEL_HPP
#define VSIP_CSL_UKERNEL_HOST_UKERNEL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/opt/ukernel/host/ukernel.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip_csl
{
namespace ukernel
{

using namespace vsip::impl::ukernel;

/// The Kernel_proxy class template is the host-side part of the User Kernel API.
///
/// Template parameters:
///
///   :Pre: Number of pre-input arguments
///   :In: Number of input arguments
///   :Out: Number of output arguments (defaults to 1)
///   :Param: Parameter set for this operation
template <unsigned int Pre, unsigned int In, unsigned int Out = 1,
	  typename Param = Empty_params>
struct Kernel_proxy
{
  typedef Param param_type;
  static unsigned int const pre_argc = Pre;
  static unsigned int const in_argc  = In;
  static unsigned int const out_argc = Out;
  static bool const extra_ok = false;

  void fill_params(param_type &) const {}
  void *get_param_stream() const { return 0;}
  length_type stack_size() const { return 4096;}
  length_type num_accel(length_type avail) const { return avail;}
};

} // namespace vsip_csl::ukernel
} // namespace vsip_csl

#endif
