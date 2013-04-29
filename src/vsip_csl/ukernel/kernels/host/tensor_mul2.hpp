/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// VSIPL++ Library: User-kernel for passing tensors (used primarily 
/// for testing).

#ifndef VSIP_CSL_UKERNEL_KERNELS_HOST_TENSOR_MUL2_HPP
#define VSIP_CSL_UKERNEL_KERNELS_HOST_TENSOR_MUL2_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip_csl/ukernel/host/ukernel.hpp>

namespace vsip_csl
{
namespace ukernel
{


/***********************************************************************
  Definitions
***********************************************************************/

class Tensor_mul2_proxy : public Kernel_proxy<0,1,1>
{
public:
  Tensor_mul2_proxy() 
    : sp_(Blocksize_sdist(4,1), Blocksize_sdist(4,1)) {}

  Stream_pattern const &in_spatt(vsip::index_type) const
  {
    return sp_;
  }

  Stream_pattern const &out_spatt(vsip::index_type) const
  {
    return sp_;
  }

private:
  Stream_pattern sp_;
};


template <>
struct Task_map<Tensor_mul2_proxy, void(float*, float*)>
{
  static char const *plugin() { return "uk_plugin/tensor_mul2_f.plg";}
};


} // namespace ukernel
} // vsip_csl

#endif // VSIP_CSL_UKERNEL_KERNELS_HOST_TENSOR_MUL2_HPP
